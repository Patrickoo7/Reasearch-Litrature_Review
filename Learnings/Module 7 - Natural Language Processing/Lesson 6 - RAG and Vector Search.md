# Lesson 6: RAG & Vector Search 🔍

**Module 7: Natural Language Processing | Lesson 6 of 7**

Build production-ready Retrieval-Augmented Generation systems - the #1 enterprise LLM use case!

---

## Why RAG? 🎯

**LLM Limitations:**
- ❌ Knowledge cutoff (training data ends at specific date)
- ❌ Hallucinations (makes up facts)
- ❌ No access to private/proprietary data
- ❌ Can't cite sources
- ❌ Expensive to update knowledge (requires retraining)

**RAG Solution:**
- ✅ Retrieve relevant documents from knowledge base
- ✅ Augment prompt with retrieved context
- ✅ Generate answer grounded in facts
- ✅ Provide source citations
- ✅ Update knowledge by adding documents (no retraining needed)

---

## 1. RAG Pipeline Overview 🏗️

```
User Query
    ↓
1. Embed query into vector
    ↓
2. Search vector database for similar documents
    ↓
3. Retrieve top-k most relevant documents
    ↓
4. Augment LLM prompt with retrieved context
    ↓
5. Generate answer using LLM
    ↓
Answer + Source Citations
```

**Key Components:**
1. **Embedding Model:** Converts text to vectors
2. **Vector Database:** Stores and searches document embeddings
3. **Retriever:** Finds relevant documents
4. **Generator:** LLM that produces final answer

---

## 2. Build Simple RAG from Scratch 🛠️

### Step 1: Prepare Documents & Embed

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Knowledge base
documents = [
    "Paris is the capital of France. It is known for the Eiffel Tower and the Louvre Museum.",
    "London is the capital of the United Kingdom. Big Ben and the Tower Bridge are famous landmarks.",
    "Tokyo is the capital of Japan. It is the world's most populous metropolitan area.",
    "Berlin is the capital of Germany. The Berlin Wall fell in 1989.",
    "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
    "Python is a popular programming language widely used for data science and machine learning.",
    "The Pacific Ocean is the largest ocean on Earth, covering more area than all land combined.",
    "Natural language processing enables computers to understand and generate human language.",
]

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

# Embed all documents
doc_embeddings = embedding_model.encode(documents, show_progress_bar=True)

print(f"Document embeddings shape: {doc_embeddings.shape}")  # (8, 384)
print(f"Each document is represented as a {doc_embeddings.shape[1]}-dimensional vector")
```

### Step 2: Semantic Search (Retrieval)

```python
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_documents(query, top_k=3):
    """Retrieve most relevant documents for a query"""
    # Embed query
    query_embedding = embedding_model.encode([query])
    
    # Compute cosine similarity between query and all documents
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Get top-k most similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'score': float(similarities[idx]),
            'index': int(idx)
        })
    
    return results

# Test retrieval
query = "What is the capital of France?"
results = retrieve_documents(query, top_k=3)

print(f"Query: {query}\n")
print("Retrieved Documents:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Similarity Score: {result['score']:.4f}")
    print(f"   Document: {result['document']}")
```

### Step 3: Augment Prompt & Generate Answer

```python
import openai

# Note: Requires OpenAI API key
# openai.api_key = "your-api-key"

def rag_generate(query, top_k=3):
    """Complete RAG pipeline: Retrieve + Augment + Generate"""
    
    # Step 1: Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, top_k=top_k)
    
    # Step 2: Build context from retrieved documents
    context = "\n\n".join([
        f"Source {i+1}: {doc['document']}"
        for i, doc in enumerate(retrieved_docs)
    ])
    
    # Step 3: Augmented prompt
    augmented_prompt = f"""Answer the question based on the context below. 
If the answer cannot be found in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
    
    # Step 4: Generate answer using LLM
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": augmented_prompt}],
        temperature=0,  # Deterministic for factual answers
        max_tokens=200
    )
    
    answer = response.choices[0].message.content.strip()
    
    return {
        'answer': answer,
        'sources': [doc['document'] for doc in retrieved_docs],
        'scores': [doc['score'] for doc in retrieved_docs]
    }

# Test RAG
query = "What is the capital of France?"
result = rag_generate(query, top_k=2)

print(f"Question: {query}\n")
print(f"Answer: {result['answer']}\n")
print("Sources:")
for i, (source, score) in enumerate(zip(result['sources'], result['scores']), 1):
    print(f"  {i}. (Score: {score:.4f}) {source}")
```

---

## 3. Vector Databases 🗄️

### FAISS (Facebook AI Similarity Search)

**Fast, efficient, local vector search**

```python
import faiss

# Prepare data
dimension = doc_embeddings.shape[1]  # 384
doc_vectors = doc_embeddings.astype('float32')  # FAISS requires float32

# Create FAISS index (L2 distance)
index = faiss.IndexFlatL2(dimension)

# Add vectors to index
index.add(doc_vectors)

print(f"Total vectors in index: {index.ntotal}")

# Search
def faiss_search(query, k=3):
    """Search using FAISS"""
    query_vector = embedding_model.encode([query]).astype('float32')
    
    # Search returns distances and indices
    distances, indices = index.search(query_vector, k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            'document': documents[idx],
            'distance': float(dist),
            'index': int(idx)
        })
    
    return results

# Test
query = "Tell me about machine learning"
results = faiss_search(query, k=3)

print(f"Query: {query}\n")
for i, result in enumerate(results, 1):
    print(f"{i}. Distance: {result['distance']:.4f}")
    print(f"   {result['document']}\n")

# Save index for later use
faiss.write_index(index, "vector_index.faiss")

# Load index
loaded_index = faiss.read_index("vector_index.faiss")
```

### ChromaDB (Easy Local Vector Database)

```python
import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create collection
collection = client.create_collection(
    name="knowledge_base",
    metadata={"description": "My knowledge base"}
)

# Add documents (ChromaDB handles embedding automatically if you provide embedding function)
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": f"document_{i}", "type": "knowledge"} for i in range(len(documents))]
)

# Query
results = collection.query(
    query_texts=["What is the capital of Japan?"],
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

print("ChromaDB Results:")
for doc, meta, dist in zip(results['documents'][0], 
                            results['metadatas'][0],
                            results['distances'][0]):
    print(f"\nDistance: {dist:.4f}")
    print(f"Metadata: {meta}")
    print(f"Document: {doc}")

# Persist
client.persist()

# Filter by metadata
filtered_results = collection.query(
    query_texts=["capital"],
    n_results=5,
    where={"type": "knowledge"}
)
```

### Pinecone (Managed Cloud Vector Database)

```python
import pinecone

# Initialize (requires API key)
pinecone.init(
    api_key="your-api-key",
    environment="us-west1-gcp"
)

# Create index
index_name = "rag-knowledge-base"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",  # cosine similarity
        shards=1
    )

# Connect to index
index = pinecone.Index(index_name)

# Prepare vectors for upsert
vectors_to_upsert = [
    (
        f"doc_{i}",  # ID
        embedding.tolist(),  # Vector
        {"text": doc, "source": f"doc_{i}"}  # Metadata
    )
    for i, (embedding, doc) in enumerate(zip(doc_embeddings, documents))
]

# Upsert vectors
index.upsert(vectors=vectors_to_upsert)

# Query
query_vector = embedding_model.encode(["What is the capital of UK?"]).tolist()

results = index.query(
    vector=query_vector[0],
    top_k=3,
    include_metadata=True
)

print("Pinecone Results:")
for match in results['matches']:
    print(f"\nScore: {match['score']:.4f}")
    print(f"ID: {match['id']}")
    print(f"Text: {match['metadata']['text']}")

# Stats
print(f"\nIndex stats: {index.describe_index_stats()}")
```

### Weaviate (Hybrid Search)

```python
import weaviate

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

# Define schema
schema = {
    "classes": [{
        "class": "Document",
        "description": "A knowledge base document",
        "vectorizer": "text2vec-transformers",
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The document content"
            },
            {
                "name": "source",
                "dataType": ["string"]
            }
        ]
    }]
}

# Create schema
client.schema.create(schema)

# Add documents
with client.batch as batch:
    for i, doc in enumerate(documents):
        batch.add_data_object(
            {
                "content": doc,
                "source": f"doc_{i}"
            },
            "Document"
        )

# Hybrid search (keyword + vector)
result = (
    client.query
    .get("Document", ["content", "source"])
    .with_hybrid(query="capital city", alpha=0.5)  # 0=keyword, 1=vector
    .with_limit(3)
    .do()
)

print("Weaviate Hybrid Search Results:")
for doc in result['data']['Get']['Document']:
    print(f"\nSource: {doc['source']}")
    print(f"Content: {doc['content']}")
```

---

## 4. Document Chunking Strategies 📄

### Fixed-Size Chunking

```python
def chunk_text_fixed(text, chunk_size=500, overlap=50):
    """Split text into fixed-size chunks with overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Move back for overlap
    
    return chunks

# Test
long_document = """
Natural language processing (NLP) is a subfield of artificial intelligence 
that focuses on the interaction between computers and human language. 
It encompasses various tasks such as text classification, named entity recognition,
machine translation, and question answering. Modern NLP leverages deep learning
techniques, particularly transformer models like BERT and GPT, to achieve 
state-of-the-art performance across many benchmarks.
""" * 10  # Repeat to make it longer

chunks = chunk_text_fixed(long_document, chunk_size=200, overlap=50)
print(f"Created {len(chunks)} chunks")
print(f"\nFirst chunk:\n{chunks[0]}...")
print(f"\nLast chunk:\n{chunks[-1]}...")
```

### Semantic Chunking (LangChain)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try these in order
)

chunks = text_splitter.split_text(long_document)

print(f"Created {len(chunks)} semantic chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1} ({len(chunk)} chars):\n{chunk}...")
```

### Token-Based Chunking

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def chunk_by_tokens(text, max_tokens=512, overlap=50):
    """Chunk text by token count (respects model limits)"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start = end - overlap
    
    return chunks

chunks = chunk_by_tokens(long_document, max_tokens=128, overlap=20)
print(f"Created {len(chunks)} token-based chunks")
```

### Sentence-Based Chunking

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def chunk_by_sentences(text, sentences_per_chunk=5, overlap=1):
    """Chunk by complete sentences"""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        chunk_sents = sentences[i:i + sentences_per_chunk]
        chunk = " ".join(chunk_sents)
        chunks.append(chunk)
    
    return chunks

chunks = chunk_by_sentences(long_document, sentences_per_chunk=3, overlap=1)
print(f"Created {len(chunks)} sentence-based chunks")
```

---

## 5. Advanced RAG with LangChain 🦜

### Basic RAG Chain

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_langchain")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",  # "stuff" all docs into context
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query
query = "What is the main topic of this document?"
result = qa_chain.run(query)
print(f"Question: {query}")
print(f"Answer: {result}")
```

### RAG with Source Citations

```python
from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

result = qa_with_sources({"question": query})

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### Conversational RAG (With Memory)

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create conversational chain
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# Multi-turn conversation
queries = [
    "What is machine learning?",
    "What are its main applications?",  # "its" refers to previous context
    "How does it differ from traditional programming?"
]

for query in queries:
    result = conv_chain({"question": query})
    print(f"\nQ: {query}")
    print(f"A: {result['answer']}")
```

---

## 6. Advanced RAG Patterns 🚀

### Multi-Query Retrieval

```python
from langchain.retrievers import MultiQueryRetriever
from langchain.llms import OpenAI

# Generate multiple query variations for better recall
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=OpenAI(temperature=0.3)
)

# Single query → Multiple variations → Better retrieval
query = "How can AI help in healthcare?"
results = multi_query_retriever.get_relevant_documents(query)

print(f"Original query: {query}")
print(f"\nRetrieved {len(results)} documents")
for i, doc in enumerate(results[:3]):
    print(f"\n{i+1}. {doc.page_content[:200]}...")
```

### HyDE (Hypothetical Document Embeddings)

```python
def hyde_retrieval(query, vectorstore):
    """
    HyDE: Generate hypothetical answer, embed it, search with it.
    Often retrieves more relevant docs than embedding the query directly.
    """
    # Generate hypothetical answer
    prompt = f"Write a detailed paragraph answering: {query}"
    
    hypothetical_answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    ).choices[0].message.content
    
    print(f"Hypothetical answer:\n{hypothetical_answer}\n")
    
    # Embed hypothetical answer (not original query!)
    hyp_embedding = embedding_model.encode([hypothetical_answer])
    
    # Search with hypothetical answer embedding
    similarities = cosine_similarity(hyp_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:5]
    
    results = [documents[idx] for idx in top_indices]
    
    return results

# Test HyDE
query = "What are the benefits of machine learning?"
hyde_results = hyde_retrieval(query, vectorstore)

print("HyDE Retrieved Documents:")
for i, doc in enumerate(hyde_results[:3]):
    print(f"\n{i+1}. {doc}")
```

### Two-Stage Retrieval (Retrieve + Rerank)

```python
from sentence_transformers import CrossEncoder

# Stage 1: Fast retrieval with bi-encoder (retrieve many candidates)
def retrieve_candidates(query, top_k=100):
    query_emb = embedding_model.encode([query])
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    candidates = [
        {'document': documents[idx], 'index': idx}
        for idx in top_indices
    ]
    
    return candidates

# Stage 2: Slow but accurate reranking with cross-encoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query, candidates, top_k=5):
    """Rerank candidates using cross-encoder"""
    # Create pairs
    pairs = [[query, cand['document']] for cand in candidates]
    
    # Score each pair
    scores = reranker.predict(pairs)
    
    # Add scores to candidates
    for cand, score in zip(candidates, scores):
        cand['rerank_score'] = float(score)
    
    # Sort by rerank score
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    
    return reranked[:top_k]

# Full two-stage retrieval
query = "What programming language is used for data science?"

# Stage 1: Retrieve 100 candidates
candidates = retrieve_candidates(query, top_k=100)
print(f"Stage 1: Retrieved {len(candidates)} candidates")

# Stage 2: Rerank to top 5
final_results = rerank_documents(query, candidates, top_k=5)
print(f"\nStage 2: Reranked to top 5:")
for i, result in enumerate(final_results, 1):
    print(f"\n{i}. Score: {result['rerank_score']:.4f}")
    print(f"   {result['document']}")
```

---

## 7. Production RAG System 🏭

### Complete RAG Application with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional

app = FastAPI(title="RAG API", version="1.0")

# Initialize components (load once at startup)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vectorstore = load_vector_db()  # Your vector DB

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    use_reranking: bool = False

class Source(BaseModel):
    document: str
    score: float
    index: int

class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float

@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: QueryRequest):
    """RAG endpoint with retrieval and generation"""
    try:
        # Retrieve documents
        docs = vectorstore.similarity_search(
            request.question, 
            k=request.top_k
        )
        
        # Optional reranking
        if request.use_reranking:
            docs = rerank_documents(request.question, docs, top_k=request.top_k)
        
        # Build context
        context = "\n\n".join([
            f"Source {i+1}: {doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        
        # Generate answer
        prompt = f"""Answer based on the context below. If you cannot answer based on the context, say so.

Context:
{context}

Question: {request.question}

Answer:"""
        
        answer = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        ).choices[0].message.content
        
        # Calculate confidence
        confidence = calculate_confidence(answer, docs)
        
        # Format sources
        sources = [
            Source(
                document=doc.page_content,
                score=doc.metadata.get('score', 1.0),
                index=i
            )
            for i, doc in enumerate(docs)
        ]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "all-MiniLM-L6-v2"}

# Run: uvicorn app:app --reload --port 8000
```

### Monitoring RAG Quality

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
rag_requests = Counter('rag_requests_total', 'Total RAG requests')
rag_latency = Histogram('rag_latency_seconds', 'RAG request latency')
retrieval_quality = Histogram('retrieval_avg_score', 'Average retrieval score')
answer_confidence = Histogram('answer_confidence', 'Answer confidence score')

def monitor_rag_quality(query, answer, retrieved_docs):
    """Monitor RAG system quality metrics"""
    metrics = {}
    
    # Retrieval metrics
    metrics['num_docs_retrieved'] = len(retrieved_docs)
    metrics['avg_similarity'] = np.mean([doc.metadata['score'] for doc in retrieved_docs])
    metrics['min_similarity'] = np.min([doc.metadata['score'] for doc in retrieved_docs])
    
    # Answer quality (LLM-as-judge)
    quality_prompt = f"""Rate the quality and accuracy of this answer from 1-5:

Question: {query}
Answer: {answer}

Rating (1-5):"""
    
    rating = int(openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": quality_prompt}]
    ).choices[0].message.content.strip())
    
    metrics['answer_quality'] = rating
    
    # Update Prometheus metrics
    retrieval_quality.observe(metrics['avg_similarity'])
    answer_confidence.observe(rating / 5.0)
    
    # Log to your logging system
    logger.info(f"RAG metrics: {metrics}")
    
    return metrics

@app.post("/rag_monitored")
async def rag_monitored(request: QueryRequest):
    start_time = time.time()
    rag_requests.inc()
    
    # ... RAG logic ...
    
    # Monitor
    metrics = monitor_rag_quality(request.question, result.answer, retrieved_docs)
    
    # Track latency
    rag_latency.observe(time.time() - start_time)
    
    return result
```

---

## Quick Reference 📖

### RAG vs Fine-Tuning

```
RAG:
✅ Up-to-date information (add new docs anytime)
✅ Cite sources (transparency)
✅ Lower cost (no training)
✅ Easy to update knowledge base
✅ Works with any LLM
❌ Slower (retrieval + generation)
❌ Quality depends on retrieval
❌ Requires vector DB infrastructure

Fine-Tuning:
✅ Faster inference (no retrieval)
✅ Learns specific style/format
✅ Can inject knowledge into model weights
❌ Knowledge becomes stale (requires retraining)
❌ No source citations
❌ Expensive to update ($$$)
❌ Catastrophic forgetting risk
```

### Vector Database Comparison

```
FAISS:
✅ Extremely fast
✅ Free, runs locally
✅ Scales to billions of vectors
❌ No built-in persistence
❌ No metadata filtering
❌ Single-machine only

ChromaDB:
✅ Easy to use (auto-embedding)
✅ Free, runs locally
✅ Metadata filtering
✅ Built-in persistence
❌ Limited scale (~millions)

Pinecone:
✅ Fully managed (no ops)
✅ Scales to billions
✅ Low latency (<100ms)
✅ Metadata filtering
❌ Paid service
❌ Vendor lock-in

Weaviate:
✅ Hybrid search (keyword + vector)
✅ GraphQL API
✅ Self-hosted or cloud
✅ Scales well
❌ More complex setup
```

---

## Practice Exercises 🏋️

### Exercise 1: Documentation Chatbot
Build a RAG system for your project's documentation:
- Load all .md files
- Chunk appropriately
- Enable semantic search
- Deploy with FastAPI

### Exercise 2: Multi-Document QA
Create a system that answers questions across multiple PDFs:
- PDF parsing
- Document deduplication
- Source attribution
- Compare different chunking strategies

### Exercise 3: Hybrid Search
Implement hybrid search combining:
- BM25 (keyword search)
- Vector similarity
- Weighted fusion
- Evaluate on benchmark dataset

---

## Key Takeaways 💡

1. **RAG = Retrieval + Augmentation + Generation** - three distinct stages
2. **Embedding model quality matters** - better embeddings = better retrieval
3. **Chunking strategy significantly impacts quality** - test different approaches
4. **Two-stage retrieval (retrieve + rerank) improves precision** dramatically
5. **RAG is more flexible than fine-tuning** for knowledge-intensive tasks
6. **Vector databases are essential** for production RAG at scale
7. **HyDE and multi-query improve recall** for complex queries
8. **Monitor retrieval quality** to catch degradation early
9. **Hybrid search (keyword + vector) often outperforms vector-only**
10. **Always cite sources** for transparency and trust

---

**Next:** [Lesson 7 - Text Generation, Evaluation & Production →](Lesson%207%20-%20Text%20Generation%20Evaluation%20and%20Production.md)
