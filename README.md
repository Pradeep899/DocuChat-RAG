# DocuChat-RAG
DocuChat — Chat with your PDFs (RAG-based Document Q&A System)
================================================================

Overview
--------
DocuChat is an end-to-end Retrieval-Augmented Generation (RAG) application that turns any PDF into an interactive, queryable knowledge source. Instead of manually searching through pages of a document, users simply upload a file and ask questions in natural language — the system retrieves the most relevant sections and generates accurate, context-grounded answers using a large language model.

The project was built to gain a practical, working understanding of how modern RAG pipelines function in production — from document ingestion and chunking, to embedding generation, vector search, and final answer synthesis — rather than treating it as a black box.

Problem it solves
------------------
Long documents (research papers, textbooks, reports, manuals) are hard to search efficiently. Traditional Ctrl+F search only matches exact keywords and misses context. DocuChat solves this by understanding the meaning behind a question and retrieving semantically relevant content, even if the exact words don't match — then uses an LLM to turn that content into a clear, direct answer.

How it works
------------
1. Document Ingestion — The uploaded PDF is parsed using PyPDFLoader and split into overlapping text chunks (1000 characters, 200 overlap) using LangChain's RecursiveCharacterTextSplitter, preserving context across chunk boundaries.

2. Embedding Generation — Each chunk is converted into a dense vector representation using the BAAI/bge-small-en-v1.5 embedding model via HuggingFace, capturing semantic meaning rather than just keywords.

3. Vector Storage & Retrieval — Embeddings are stored in a Chroma vector database. When a question is asked, DocuChat uses Maximal Marginal Relevance (MMR) search (k=4, fetch_k=10, lambda_mult=0.5) to fetch chunks that are both highly relevant and non-redundant.

4. Answer Generation — Retrieved chunks are injected into a structured prompt template and passed to Google's Gemini LLM (gemini-3.5-flash) via LangChain. The model is explicitly instructed to answer only from the provided context, and to say so honestly if the answer isn't present — reducing hallucination.

5. Interface — The entire pipeline is wrapped in a Streamlit web app with a chat-style UI, live document processing status, and expandable "source" panels so users can verify exactly which part of the document an answer came from.

Tech stack
----------
- Frontend / App Framework: Streamlit
- Orchestration: LangChain
- PDF Parsing: PyPDFLoader
- Text Splitting: RecursiveCharacterTextSplitter
- Embedding Model: BAAI/bge-small-en-v1.5 (HuggingFace)
- Vector Database: ChromaDB
- Retrieval Method: MMR (Maximal Marginal Relevance)
- LLM: Google Gemini (gemini-3.5-flash) via ChatGoogleGenerativeAI
- Environment Management: python-dotenv

Key features
------------
- Upload any PDF directly through the browser, no manual preprocessing required
- Natural language, chat-style question answering
- Source attribution for every answer — full transparency into retrieved content
- Semantic search that understands meaning, not just keywords
- Grounded generation that avoids hallucinated answers outside the document
- Clean, responsive, user-friendly UI

What I learned
---------------
Building DocuChat gave me hands-on experience with the practical engineering decisions behind RAG systems — how chunk size and overlap affect retrieval quality, why diversity-aware retrieval (MMR) outperforms plain similarity search for avoiding repetitive context, how to structure prompts to keep an LLM factually grounded, and how to turn a working script into an actual usable product with a proper interface.

This project is part of a broader portfolio of Generative AI and NLP projects I've been building while developing my machine learning skill set.
