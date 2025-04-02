# GraphRAG with Neo4j

A powerful implementation of Retrieval-Augmented Generation (RAG) using Neo4j graph database and local LLM models through Ollama. This project combines vector search, graph-based retrieval, and hybrid approaches to enhance question-answering capabilities.

## Features

- Multiple retrieval strategies:
  - Vector-based retrieval
  - Vector + Cypher hybrid retrieval
  - Full-text + Vector hybrid retrieval
  - Text-to-Cypher conversion for graph queries
- Local LLM integration using Ollama
- Graph-based knowledge representation
- Flexible entity relationships and properties

## Prerequisites

- Python 3.8+
- Neo4j Database
- Ollama server running locally
- Required Python packages (see `pyproject.toml`)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
uv pip install -r pyproject.toml
```
3. Set up environment variables in `.env`:
```
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
OLLAMA_SERVER=your_ollama_server_url
LLM_MODEL=your_preferred_model
EMBEDDING_MODEL=your_embedding_model
```

## Project Structure

### Main Components

1. `graph_rag.py`
   - Main implementation of GraphRAG with different retrieval strategies
   - Includes Neo4j schema definition and various retriever implementations
   - Supports vector, hybrid, and graph-based retrieval methods

2. `ollama_local.py`
   - Configuration for local LLM and embedding models
   - Sets up Ollama integration for text generation and embeddings

3. `custom_ollama_llm.py`
   - Custom implementation of Ollama LLM integration
   - Handles model interactions and response processing

### Neo4j Schema

The project uses the following node types:
- `Chunk`: Text chunks with embeddings
- `Concept`: Domain concepts and their relationships
- `Document`: Source documents
- `Event`: Event entities
- `Object`: Object entities
- `Person`: Person entities
- `Technology`: Technology-related entities

Relationships include:
- `PART_OF`: Connects chunks to documents
- `HAS_ENTITY`: Links chunks to various entity types
- `HAS`: Concept relationships
- `USES`: Usage relationships between concepts

## Usage Examples

### Basic Vector Retrieval
```python
vec_retriever = VectorRetriever(
    driver,
    index_name="chunkVector",
    embedder=embedding_provider,
    return_properties=["id", "text"],
)
rag = GraphRAG(retriever=vec_retriever, llm=llm)
response = rag.search(query_text="What is GPT?", retriever_config={"top_k": 5})
```

### Hybrid Retrieval with Cypher
```python
hybrid_cypher_retriever = HybridCypherRetriever(
    driver=driver,
    vector_index_name="chunkVector",
    fulltext_index_name="chunkText",
    retrieval_query=your_cypher_query,
    embedder=embedding_provider,
)
rag = GraphRAG(retriever=hybrid_cypher_retriever, llm=llm)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.