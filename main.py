
import os
import chromadb
from typing import TypedDict
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from fastapi import FastAPI
from pydantic import BaseModel, Field

MOCK_LLM = os.getenv("MOCK_LLM", "1")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)

prompt_template = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
You must use the retrieved Zepto policy documents as the only source of policy information.

TASK:
Answer the customer's question using only the relevant information from the retrieved context.

FORMAT:
Return a direct and clear answer.

LENGTH:
Keep the answer between 2 and 4 sentences when possible.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies, prices, timings, refunds, delivery rules, or support services.

FEW-SHOT EXAMPLE:
Question: What is the delivery fee for orders below INR 149?
Context: Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.
Answer: Orders below INR 149 have a flat INR 25 standard delivery fee.

CUSTOMER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
"""

class AskRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)

class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float

def classify_intent(state):
    query = state["query"].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    state["intent"] = "policy_question" if any(keyword in query for keyword in keywords) else "general_question"

    return state

def retrieve_and_answer(state):
    query = state["query"]

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(3, collection.count())
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    if not documents:
        return {
            **state,
            "answer": "No relevant Zepto policy information was found.",
            "sources": [],
            "confidence": 0.0
        }

    top_chunk = documents[0]
    snippet = top_chunk[:200]

    if MOCK_LLM == "1":
        answer = f"Based on the retrieved context: {snippet}"
    else:
        context = "\n\n".join(documents)
        prompt = prompt_template.format(
            question=query,
            context=context
        )
        answer = snippet

    response = AnswerResponse(
        answer=answer,
        sources=ids,
        confidence=1.0
    )

    return {
        **state,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }

def direct_answer(state):
    response = AnswerResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )

    return {
        **state,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }

def route_question(state):
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

builder = StateGraph(GraphState)

builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.set_entry_point("classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()

app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest):
    result = graph.invoke({
        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    })

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )
