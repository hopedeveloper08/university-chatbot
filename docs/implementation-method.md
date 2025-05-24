## Overview

This project aims to implement a Retrieval-Augmented Generation (RAG) system in multiple phases, starting with a **Simple RAG** architecture. The primary goal is to build a modular and extensible pipeline that combines document preprocessing, semantic search using vector databases, and large language model (LLM) responses.

The system consists of four main components:

1. **Chunking**
2. **Embedding**
3. **Semantic Search**
4. **LLM Response Generation**

Future versions will incrementally add advanced capabilities to improve retrieval precision and response quality.


## Phase 1: Simple RAG

### 1. Document Preprocessing & Chunking

The raw text documents are first preprocessed and divided into chunks of fixed length. A special rule is applied during chunking: if a chunk ends in the middle of a sentence, it is extended until the sentence reaches its end (i.e., until a period `.` is found). This ensures semantic coherence in each chunk.

### 2. Embedding with ParsBERT

Each chunk is converted into a dense vector representation using **ParsBERT**, a transformer-based language model optimized for Persian. These embeddings are stored in a vector store for efficient similarity search.

### 3. Vector Storage using FAISS

We use **FAISS** (Facebook AI Similarity Search) as our vector store. It allows fast nearest neighbor search for high-dimensional vectors, making it ideal for our semantic search component.

### 4. Retrieval and Prompt Construction

Once the vector store is ready, a user query is embedded and semantically matched against stored vectors. The top relevant chunks are retrieved and passed into a prompt template that feeds into the LLM.

### 5. Response Generation with Gemma 3:4b

We employ the **Gemma 3:4b** LLM to generate final answers. The model receives the retrieved chunks and user query as context in a structured prompt. This setup completes the RAG pipeline, delivering context-aware responses.


## Phase 2 and Beyond: Planned Enhancements

The following enhancements are planned for upcoming versions:

### 2.1 Metadata-Aware Retrieval

- Attach metadata (e.g., source, document type, date) to each chunk.
- Enhance retrieval by filtering or prioritizing based on metadata fields.

### 2.2 QA-Based Data Augmentation

- Automatically generate question-answer pairs from documents and chunks.
- Store and retrieve them in a dedicated vector space.
- Enable hybrid retrieval of both facts and direct answers.

### 2.3 Context-Aware Retrieval Expansion

- During retrieval, fetch adjacent chunks (previous and next) to enrich context.
- Improve completeness and coherence of the retrieved knowledge.
