# Table of contents

- [**Simple RAG**](#simple-rag)
- [**Semantic Chunking**](#semantic-chunking-alternative-to-fixed-length-chunking)
- [**Context Enriched**](#context-enriched)
- [**Chunking with Metadata**](#chunking-with-metadata)
- [**Document Augmentation with Question Generation**](#document-augmentation-with-question-generation)
- [**Query Transformation**](#query-transformation)

# Simple RAG 

RAG is an approach combining information retrieval techniques and generative language models with the aim of generating answers to questions given our knowledge base. The simplest idea for implementing RAG architecture is as follows.

### Steps:  

1. **Data Ingestion**:  
    Load and preprocess the raw text data to prepare it for further processing.

2. **Chunking**:  
    Split the text into smaller chunks of equal character length with a fixed amount of overlap between consecutive chunks. This improves retrieval effectiveness.

3. **Embedding Creation**:  
    Generate numerical vector representations (embeddings) for each chunk using a suitable embedding model.

4. **Semantic Search**:  
    - Convert the user query into an embedding using the same model.
    - Compute the cosine similarity between the query embedding and all text chunk embeddings.
    - Rank the chunks based on similarity scores.

5. **Response Generation**:  
    Select the top-ranked chunks and pass them along with the user query to a language model, which generates the final response.


# Semantic Chunking (Alternative to Fixed-Length Chunking)

Unlike fixed-length chunking, semantic chunking divides the text based on content similarity between sentences.

### Steps:

1. **Sentence Splitting**:  
   Split the text into individual sentences.

2. **Initial Embedding**:  
   Generate embeddings for each sentence using a suitable embedding model.

3. **Cosine Similarity Calculation**:  
   Compute cosine similarity between each pair of consecutive sentences.

4. **Breakpoint Detection**:  
   Identify chunk boundaries using one of the following thresholding methods:
   - **Percentile**: `percentile(similarities, threshold)`
   - **Standard Deviation**: `mean - (threshold * std_dev)`
   - **Interquartile Range (IQR)**: `q1 - 1.5 * (q3 - q1)`

5. **Chunk Formation**:  
   Break the text into semantic chunks at the detected breakpoints.

6. **Re-embedding**:  
   Generate embeddings for each new chunk.

7. **Continue with Simple RAG Pipeline**:  
   Perform semantic search and response generation as in the Simple RAG method.

# Context Enriched

In the basic Simple RAG method, the top-K most similar chunks are retrieved and passed to the language model. However, this approach may lead to incomplete context, as relevant information might be spread across neighboring chunks.

### Problem:
- Retrieved chunks may lack surrounding context.
- The model may miss important information present in adjacent chunks.

### Solution:
For each of the top-K retrieved chunks:
- Also include their neighboring chunks (e.g., previous and next based on their index).
- This provides more complete and coherent context to the model during response generation.

# Chunking with Metadata

In standard chunking, important contextual information may be lost. To address this, metadata such as concise titles or tags can be added to each chunk to improve retrieval performance.

### Benefits:
- Enhances the semantic relevance of retrieved chunks.
- Helps the model better understand the context of each chunk.

### How it Works:
- Add a short, meaningful title or other metadata to each chunk during preprocessing.
- During semantic search, compute similarity not only based on chunk content but also on metadata.
- Improves the chance of retrieving more relevant and complete information.

# Document Augmentation with Question Generation

Sometimes, documents contain ambiguity or lack clear intent. Generating questions from the documents can help clarify the content and improve retrieval.

### Purpose:
- Makes the meaning and usage of the content more explicit.
- Enhances the semantic richness of the knowledge base.

### Method:
- Use large language models (LLMs) to automatically generate question-answer pairs from the documents.
- Store the generated Q&A alongside the original chunks.
- During retrieval, match user queries not only with document chunks but also with generated questions to increase recall and relevance.

# Query Transformation

Enhancing the user query through prompt engineering techniques can significantly improve retrieval quality.

### Techniques:
1. **Query Rewriting**: Reformulate the original query for clarity and precision.
2. **Query Expansion**: Add relevant keywords or context to enrich the query.
3. **Query Decomposition**: Break down complex queries into simpler sub-questions.

### Implementation:
- These techniques can be automated using an intelligent agent (e.g., an LLM) before performing the retrieval step.
- The transformed query helps in retrieving more accurate and contextually relevant chunks.
