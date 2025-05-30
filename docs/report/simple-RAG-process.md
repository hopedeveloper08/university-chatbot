# Preprocessing and Integration 

## Overview
The collected documents have been successfully integrated and undergone initial preprocessing to ensure consistency and cleanliness of the textual data.

## Preprocessing Steps
The following steps were applied during the initial preprocessing phase:

1. **Whitespace Removal**  
   - Extra and unnecessary whitespaces were removed to normalize text formatting.

2. **Character Normalization**  
   - Arabic characters were replaced with their Persian equivalents (e.g., "ي" → "ی", "ك" → "ک").

3. **Punctuation Normalization**  
   - Quotation marks were replaced with standardized characters.
   - Repeated and meaningless punctuation characters were removed (e.g., "!!!", "??", "...", etc.).

# Vector Store Creation Phase

## Overview

This phase focuses on the implementation of three core modules required to build the vector store component of the Retrieval-Augmented Generation (RAG) system:

1. **Chunking**
2. **Embedding**
3. **Vector Store Management**

Each module is designed with configurability and scalability in mind, enabling flexibility in processing various types of textual data using transformer-based language models.

## Chunking Module

### Purpose:
To preprocess and divide raw text data into manageable and semantically coherent chunks for downstream embedding and storage.

### Implementation Details:
- **Input**: Path to raw text files.
- **Processing**:
  - Files are loaded and processed using `RecursiveCharacterTextSplitter`.
  - Chunking is governed by configurable parameters:
    - `chunk_size`
    - `chunk_overlap`
    - `separators`
- **Output**: A list of `Document` objects containing the text chunks.

## Embedding Module

### Purpose:
To convert text chunks into vector representations using a pre-trained language model.

### Implementation Details:
- **Model Selection**: The embedding model is specified via the configuration file.
- **Device Adaptation**: 
  - If a GPU is available, it is utilized; otherwise, the CPU is used.
- **Model Used in v1**: `ParsBERT` is employed for embedding text in the initial version.
- **Embedding Loader**: A utility function loads and prepares the embedding model based on configuration and device availability.

## Vector Store Module

### Purpose:
To store and retrieve vector representations of text chunks for similarity-based document retrieval.

### Functions Implemented:
1. **Vector Store Creation**:
   - Takes in `documents`, the embedding model, and a save path.
   - Generates and stores the vector index.
2. **Vector Store Loader**:
   - Loads a pre-existing vector store from disk using its path and the same embedding model.

# Retrieval & Generation Phase

## Overview

This phase focuses on implementing a **retrieval-augmented generation (RAG)** pipeline. It combines a retriever module that fetches relevant information from a vector store with a generator module that formulates answers based on a prompt template and a language model.

## Components

### 1. **Retriever Module**

* Initializes a retriever from an existing **vector database**.
* Responsible for fetching top relevant documents in response to user queries.
* Acts as the first step in the pipeline to provide context for generation.

### 2. **Generator Module**

* Composed of:

  * **Prompt Template**: Defines the structure of input to the language model.
  * **Language Model**: In version 1, we use [`Ollama`](https://ollama.com) with the `Gemma3:4b` model as the LLM backend.
* Generates the final response by combining the query and retrieved documents using the prompt template.

### 3. **Question-Answer Chain**

* A **pipeline** that links the retriever and generator components.
* When the user submits a query:

  1. The retriever fetches relevant context.
  2. The generator uses the context and prompt to produce a final answer.
* This chain abstracts the process into a seamless **Q\&A experience** for the end user.
