
# Version 1.0 - Simple RAG

## Overview

This project aims to implement a Retrieval-Augmented Generation (RAG) system in multiple phases, starting with a **Simple RAG** architecture. The primary goal is to build a modular and extensible pipeline that combines document preprocessing, semantic search using vector databases, and large language model (LLM) responses.

The system consists of four main components:

1. **Chunking**
2. **Embedding**
3. **Semantic Search**
4. **LLM Response Generation**

Future versions will incrementally add advanced capabilities to improve retrieval precision and response quality.

## Steps

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

# Version 1.1 - Clean input - Streams output 

### 1. Modular Refactoring

The system architecture has been restructured into two clearly separated modules:

- **Offline Module (`prepare_data`)**: Responsible for data preprocessing, chunking, and embedding.
    
- **Online Module (`rag_system`)**: Handles real-time query processing and response generation.
    

Each module exposes a single high-level interface:

- `prepare_data()`: Preprocesses the raw documents, performs chunking, and stores embeddings in the vector database.
    
- `build_rag_chain()`: Loads the vector store and constructs the full Retrieval-Augmented Generation pipeline for inference.
    

This modular design improves maintainability and enables independent testing of the offline and online components.

### 2. Enhanced Data Cleaning 

The data pipeline now includes more rigorous preprocessing steps. These enhancements help reduce noise and improve semantic quality:

- Removal of irrelevant symbols, HTML tags, and redundant whitespace
- Normalization of Persian characters (e.g., ی to ي, ک to ك)
- Sentence segmentation based on Persian punctuation and linguistic patterns
- Filtering of low-information or overly short segments

This results in higher-quality chunks for embedding and retrieval.

### 3. Real-Time Streaming Response

The response generation process now supports **streaming output**. Once the retrieval and prompt preparation are complete, the LLM generates the response incrementally. This leads to faster perceived latency and improves user experience, especially for long-form answers.