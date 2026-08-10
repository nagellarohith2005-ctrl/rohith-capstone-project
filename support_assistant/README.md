# Module 3 — Support Assistant

## Overview

This module implements a small GenAI-powered customer support assistant for Zepto.

The system uses a Retrieval-Augmented Generation (RAG) pipeline with:

* Zepto policy documents
* Sentence Transformers for local embeddings
* ChromaDB for vector storage and retrieval
* LangGraph for intent routing
* Pydantic for structured responses
* FastAPI for the API service
* Docker for local containerization

The required graded implementation runs in deterministic offline mock mode. No LLM API key or external LLM service is required.

---

## Project Structure

```text
support_assistant/
│
├── module_3.ipynb
├── main.py
├── Dockerfile
├── requirements.txt
├── README.md
│
└── docs/
    ├── doc_01.txt
    ├── doc_02.txt
    ├── doc_03.txt
    ├── doc_04.txt
    ├── doc_05.txt
    ├── doc_06.txt
    ├── doc_07.txt
    └── doc_08.txt
```

---

## Technologies Used

* Python
* Pandas
* Sentence Transformers
* `all-MiniLM-L6-v2`
* ChromaDB
* LangGraph
* Pydantic
* FastAPI
* Uvicorn
* Docker

---

# Task 1 — Document Ingestion and Embeddings

The `docs/` directory contains eight Zepto policy documents covering:

1. Delivery Policy
2. Returns and Refunds
3. Membership Tiers
4. Order Tracking
5. Order Cancellation
6. Damaged or Missing Items
7. Gift Cards
8. Customer Support Hours

Each document is loaded and embedded using the open-source:

```text
all-MiniLM-L6-v2
```

The generated embeddings are stored in a ChromaDB collection named:

```text
zepto_policies
```

The notebook uses ChromaDB's in-memory client for the offline implementation, so no API key or external vector database is required.

---

# Task 2 — Structured Prompt

The prompt follows the required:

```text
Role → Context → Task → Format → Length
```

structure.

It also includes:

* A negative constraint
* A few-shot example
* Grounding instructions
* Retrieved policy context
* Customer question

The prompt explicitly prevents the assistant from answering with information that is not contained in the retrieved Zepto policy documents.

---

# Task 3 — LangGraph Workflow

The application uses a LangGraph `StateGraph` with three nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The workflow is:

```text
                   ┌─────────────────────┐
                   │  classify_intent    │
                   └──────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
             policy_question      general_question
                    │                   │
                    ▼                   ▼
       ┌────────────────────┐   ┌─────────────────┐
       │ retrieve_and_answer │   │  direct_answer  │
       └────────────────────┘   └─────────────────┘
```

### Intent Classification

In the required mock mode, the query is classified using keywords.

The policy keywords are:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

If one of these keywords is present, the query is classified as:

```text
policy_question
```

Otherwise it is classified as:

```text
general_question
```

---

# Task 4 — Structured Output

The final response is validated using a Pydantic model.

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}
```

The fields are:

### answer

The final response provided to the customer.

### sources

The document or chunk IDs used for the answer.

For general questions this list is empty.

### confidence

A value between `0` and `1`.

In mock mode the confidence is deterministically set to:

```text
1.0
```

---

# Mock LLM Mode

The graded implementation uses:

```text
MOCK_LLM=1
```

or leaves `MOCK_LLM` unset.

In this mode there is no external LLM API call.

For policy questions, the system:

1. Classifies the query.
2. Creates a local embedding.
3. Searches the ChromaDB collection.
4. Retrieves the top 3 documents/chunks.
5. Takes the most relevant chunk.
6. Generates a deterministic response.

The mock response follows:

```text
Based on the retrieved context: <top chunk snippet>
```

For general questions, the system returns:

```text
I can only answer questions about Zepto policies right now.
```

---

# RAG Architecture

The complete pipeline is:

```text
Documents
   │
   ▼
Ingestion
   │
   ▼
Chunking
   │
   ▼
Sentence Transformer
all-MiniLM-L6-v2
   │
   ▼
ChromaDB
zepto_policies
   │
   ▼
User Query
   │
   ▼
classify_intent
   │
   ├── policy_question
   │       │
   │       ▼
   │   retrieve_and_answer
   │       │
   │       ▼
   │   Top 3 ChromaDB chunks
   │       │
   │       ▼
   │   Generation
   │
   └── general_question
           │
           ▼
       direct_answer
```

### Ingestion

The policy documents are stored in:

```text
support_assistant/docs/
```

The notebook loads all eight `.txt` files.

### Embedding

`all-MiniLM-L6-v2` converts the documents and user queries into numerical vectors.

### Retrieval

The `retrieve_and_answer` LangGraph node performs vector similarity search against the:

```text
zepto_policies
```

ChromaDB collection.

The top three most similar chunks are retrieved.

### Generation

For policy questions, `retrieve_and_answer` generates the final response using the retrieved context.

For general questions, `direct_answer` provides the fixed mock response.

---

# MOCK_LLM Toggle

The `MOCK_LLM` environment variable controls the generation behavior.

### Default

```text
MOCK_LLM=1
```

No external LLM is called.

### Optional real LLM

```text
MOCK_LLM=0
```

In the optional implementation, the LLM can be used for:

* Intent classification
* Policy answer generation
* General question generation

The retrieval process remains local and continues to use Sentence Transformers and ChromaDB.

The real-LLM path is optional and is not required for the graded baseline.

---

# Task 5 — FastAPI

The FastAPI application is implemented in:

```text
main.py
```

The API provides:

```text
POST /ask
```

### Request

```json
{
  "query": "What is the delivery fee for orders below INR 149?"
}
```

### Response

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials...",
  "sources": [
    "doc_01"
  ],
  "confidence": 1.0
}
```

---

# Example API Calls

Run the application with:

```bash
uvicorn main:app --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

### Example 1 — Policy Question

Request:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d "{\"query\":\"What is the delivery fee for orders below INR 149?\"}"
```

Example response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": [
    "doc_01"
  ],
  "confidence": 1.0
}
```

### Example 2 — General Question

Request:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d "{\"query\":\"What is the capital of India?\"}"
```

Example response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

# Task 6 — Docker

The project includes a `Dockerfile` for local containerization.

Build the image:

```bash
docker build -t zepto-support-assistant .
```

Run the container:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

The API will then be available at:

```text
http://localhost:7860
```

The container runs FastAPI using Uvicorn.

---

# Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

---

# Offline Graded Baseline

The project is designed to work without:

* LLM API keys
* Cloud vector databases
* Internet access to an LLM provider
* Paid services
* External model APIs

The embeddings are generated locally using:

```text
all-MiniLM-L6-v2
```

and the required LLM behavior is handled deterministically through mock logic.

---

# Conclusion

This module demonstrates a complete offline RAG-based support assistant:

```text
Zepto Documents
      ↓
Local Embeddings
      ↓
ChromaDB
      ↓
LangGraph Intent Router
      ↓
Policy Retrieval
      ↓
Grounded Answer
      ↓
Pydantic Validation
      ↓
FastAPI
      ↓
Docker
```

The implementation satisfies the required offline baseline for document ingestion, embeddings, retrieval, LangGraph orchestration, structured output, FastAPI serving, and Docker containerization.
