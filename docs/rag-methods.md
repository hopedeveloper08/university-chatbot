# Table of contents

- [**Simple RAG**](#simple-rag)
- [**Semantic Chunking**](#semantic-chunking-alternative-to-fixed-length-chunking)
- [**Context Enriched**](#context-enriched)
- [**Chunking with Metadata**](#chunking-with-metadata)
- [**Document Augmentation with Question Generation**](#document-augmentation-with-question-generation)
- [**Query Transformation**](#query-transformation)
- [**Reranking**](#reranking)
- [**Relevant Segment Extraction**](#relevant-segment-extraction)
- [**Contextual Compression**](#contextual-compression)
- [**Feedback Loop**](#feedback-loop)
- [**Adaptive Retrieval**](#adaptive-retrieval)
- [**Self RAG**](#self-rag)

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

# Improvement

## Semantic Chunking

Semantic chunking splits text based on the semantic similarity between sentences, unlike fixed-length chunking which relies on character count.

First, the text is divided into individual sentences. Each sentence is then converted into an embedding using a suitable model. Cosine similarity is calculated between each pair of consecutive sentences to detect potential breakpoints.

Breakpoints can be determined using methods such as:
- Percentile thresholding
- Standard deviation rule: `mean - (threshold × std_dev)`
- Interquartile range: `q1 - 1.5 × (q3 - q1)`

Based on these breakpoints, the text is segmented into semantically meaningful chunks. These new chunks are then re-embedded, and the standard RAG pipeline continues with semantic search and response generation.


## Context Enriched

In the basic Simple RAG method, the top-K most similar chunks are retrieved and passed to the language model. However, this approach may lead to incomplete context, as relevant information might be spread across neighboring chunks.

### Problem:
- Retrieved chunks may lack surrounding context.
- The model may miss important information present in adjacent chunks.

### Solution:
For each of the top-K retrieved chunks:
- Also include their neighboring chunks (e.g., previous and next based on their index).
- This provides more complete and coherent context to the model during response generation.

## Chunking with Metadata

In standard chunking, important contextual information may be lost. To address this, metadata such as concise titles or tags can be added to each chunk to improve retrieval performance.

### Benefits:
- Enhances the semantic relevance of retrieved chunks.
- Helps the model better understand the context of each chunk.

### How it Works:
- Add a short, meaningful title or other metadata to each chunk during preprocessing.
- During semantic search, compute similarity not only based on chunk content but also on metadata.
- Improves the chance of retrieving more relevant and complete information.

## Document Augmentation with Question Generation

Sometimes, documents contain ambiguity or lack clear intent. Generating questions from the documents can help clarify the content and improve retrieval.

### Purpose:
- Makes the meaning and usage of the content more explicit.
- Enhances the semantic richness of the knowledge base.

### Method:
- Use large language models (LLMs) to automatically generate question-answer pairs from the documents.
- Store the generated Q&A alongside the original chunks.
- During retrieval, match user queries not only with document chunks but also with generated questions to increase recall and relevance.

## Query Transformation

Enhancing the user query through prompt engineering techniques can significantly improve retrieval quality.

### Techniques:
1. **Query Rewriting**: Reformulate the original query for clarity and precision.
2. **Query Expansion**: Add relevant keywords or context to enrich the query.
3. **Query Decomposition**: Break down complex queries into simpler sub-questions.

### Implementation:
- These techniques can be automated using an intelligent agent (e.g., an LLM) before performing the retrieval step.
- The transformed query helps in retrieving more accurate and contextually relevant chunks.

## Reranking

Reranking is used to improve the quality of retrieved results by applying a second-level ranking after the initial similarity-based retrieval.

### How It Works:
- After retrieving and ranking chunks based on cosine similarity, a second ranking is performed using more advanced techniques.

### Techniques:
1. **LLM-based Reranking**: Use a large language model to evaluate and reorder the top results based on relevance to the query.
2. **Keyword-based Reranking**: Score and rank the chunks based on the presence and importance of individual query keywords.
3. **FlashRank**: A powerful, lightweight and fast algorithm.

This step refines the final set of retrieved chunks for better response quality.

## Relevant Segment Extraction

Sometimes, retrieved chunks are too long and contain unnecessary information. This can confuse the language model or reduce its ability to extract the needed details.

To address this, Relevant Segment Extraction is used. Instead of passing the entire retrieved chunk to the model, only the part that is most relevant to the user query is extracted and provided to the model. This improves both the precision of the response and the efficiency of the system.

## Contextual Compression

Relevant Segment Extraction can sometimes miss important information or fail to clearly separate relevant from irrelevant parts. To overcome this, contextual compression is used.

Instead of manually selecting relevant parts, the retrieved chunks are passed to a language model, which acts as an intelligent filter. The model compresses the content while preserving the information most relevant to the user query.

There are several strategies for contextual compression:
1. Selecting only the relevant parts.
2. Summarizing the retrieved chunk with a focus on the query.
3. Extracting or generating new content specifically related to the query.

This method helps reduce noise while retaining useful information for better response generation.

## Feedback Loop

Integrating a feedback loop into the Simple RAG architecture introduces adaptability and continuous improvement over time.

By storing the outcomes of user interactions and incorporating that feedback into future similarity scoring or reranking, the system can gradually enhance the accuracy of its retrievals.

This dynamic approach helps fine-tune the system based on real user input, making it more effective and personalized over time.

## Adaptive Retrieval

Adaptive retrieval tailors the retrieval strategy based on the type of user query. By classifying queries into categories, the system can apply specialized techniques for more effective information retrieval.

### Query Categories and Strategies:

- **Factual Queries**:  
  Focus on precision. Use a language model to refine and enrich the query with specific keywords before retrieval.
- **Analytical Queries**:  
  Cover all aspects of the query. Use a language model to decompose complex queries into simpler sub-questions and retrieve relevant content for each.
- **Opinion-Based Queries**:  
  Use the query to extract or generate a list of relevant perspectives or opinions using a language model.
- **Contextual Queries**:  
  Provide background summaries related to the query to establish a proper context before generating the final response.

This approach increases the adaptability and accuracy of the RAG system across different user intents.

## Self RAG

Self RAG delegates decision-making to the language model itself, allowing it to dynamically control different stages of the RAG process based on the query and retrieved content.

The model performs the following tasks:

- Determine whether external information is needed for the given query.
- Assess whether retrieved content is relevant to the query.
- Evaluate if the generated response aligns well with the retrieved content.
- Decide whether the generated response is appropriate and useful for answering the original query.

This approach allows for more intelligent, query-aware handling of retrieval and generation, leading to improved accuracy and reliability in responses.

## Proposition Chunking

Proposition chunking transforms each chunk into a set of atomic propositions, making the content more precise and structured for retrieval and reasoning.

Each atomic proposition should:
- Express a single, clear fact.
- Be understandable without external context.
- Use full names instead of ambiguous references or pronouns.
- Focus on one topic, avoiding conjunctions and multiple clauses.

Once atomic propositions are generated, a separate model can be used to evaluate their quality. Only high-quality propositions are retained for use in the retrieval and generation process.

## Multi-Modal RAG

When the data includes images alongside text, Multi-Modal RAG can be used to enable image-based retrieval and reasoning.

By applying captioning techniques, each image is converted into descriptive text. These captions are then embedded and stored along with textual data, enabling the system to retrieve and reason over both text and images based on the user query.

This approach extends the capabilities of RAG to support richer, multi-modal information sources.
