# ChromaDB – Semantic Search & Vector Database Practice

A hands-on practice project exploring **ChromaDB, text embeddings, vector storage, and semantic search** using Python and Sentence Transformers.

This project demonstrates how raw text documents can be converted into numerical embeddings, stored in a vector database, and retrieved based on semantic similarity rather than simple keyword matching.

## Overview

The objective of this practice session is to understand the fundamentals of vector databases and build a working document retrieval pipeline.

The application imports text documents, splits them into smaller chunks, generates embeddings, stores them in ChromaDB, and retrieves the most relevant content based on user queries.

## Key Features

- **Document Importation:** Load text documents from local files.
- **Text Chunking:** Split large documents into smaller searchable segments.
- **Embedding Generation:** Convert text into numerical vector representations.
- **Vector Storage:** Store documents, embeddings, and metadata in ChromaDB.
- **Semantic Search:** Retrieve contextually relevant documents using vector similarity.
- **Top-K Retrieval:** Fetch the most relevant document chunks.
- **Similarity Ranking:** Rank retrieved results based on cosine similarity.
- **Best Match Identification:** Identify the highest-scoring search result.
- **Persistent Storage:** Retain indexed documents using ChromaDB's persistent client.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| ChromaDB | Vector database and similarity search |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Lightweight embedding model |
| Pathlib | Document file handling |

## Project Structure

```text
ChromaDB-Practice/
│
├── data/
│   └── sample.txt
│
├── chroma_db/
│   └── Persistent vector database
│
├── main.py
│
├── requirements.txt
│
└── README.md
```

## Installation & Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ChromaDB-Practice
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python main.py
```

## How It Works

The application follows a basic vector retrieval pipeline:

```text
          Input Document
                |
                v
        Document Importation
                |
                v
           Text Chunking
                |
                v
       Embedding Generation
                |
                v
       ChromaDB Vector Store
                |
                v
          User Query
                |
                v
       Query Embedding
                |
                v
        Similarity Search
                |
                v
         Top-K Retrieval
                |
                v
       Similarity Ranking
                |
                v
         Best Match Result
```

## Understanding Similarity Search

The project uses cosine distance to identify semantically related document chunks.

For a cosine-based ChromaDB collection:

**Cosine Similarity = 1 − Cosine Distance**

A higher similarity score generally indicates a closer semantic relationship between the query and the retrieved document.

The application displays:

- Search query
- Retrieved document content
- Source filename
- Chunk number
- Similarity score
- Distance score
- Highest-ranked matching result

## Sample Testing Queries

The knowledge base contains 50 paragraphs covering Artificial Intelligence and related technologies.

Example queries:

1. What is Artificial Intelligence?
2. How does Machine Learning work?
3. What is the difference between supervised and unsupervised learning?
4. How do Transformers work?
5. Explain Retrieval Augmented Generation.
6. How does ChromaDB perform similarity search?
7. What are sentence embeddings?
8. What is the purpose of metadata filtering?
9. How does hybrid search improve retrieval?
10. What is Self-Corrective RAG?

## Learning Outcomes

Through this implementation, I explored:

- Fundamentals of vector databases.
- Differences between keyword search and semantic search.
- Text-to-vector conversion using embedding models.
- Document ingestion and chunking strategies.
- Persistent vector storage.
- Cosine similarity and distance calculations.
- Top-K retrieval and relevance ranking.
- The foundational retrieval component of RAG architectures.

## Future Improvements

- [ ] PDF and DOCX document ingestion.
- [ ] Recursive and semantic text chunking.
- [ ] Metadata-based filtering.
- [ ] Configurable similarity thresholds.
- [ ] Hybrid search using BM25 and vector retrieval.
- [ ] Cross-encoder reranking.
- [ ] Flask-based document upload interface.
- [ ] Integration with an LLM for complete RAG implementation.
- [ ] Retrieval evaluation using Precision@K and Recall@K.

## Purpose

This repository is part of my practical learning journey in **Generative AI, Retrieval Augmented Generation, and Agentic AI**.

The focus is on understanding the underlying implementation of retrieval systems before progressing toward advanced RAG pipelines and production-oriented AI applications.

---

**Learning by building. Understanding by implementing.**
