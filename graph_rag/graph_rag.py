import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from ollama_local import llm, embedding_provider
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.retrievers import Text2CypherRetriever

load_dotenv()

# setup neo4j driver
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# VectorRetriever
vec_retriever = VectorRetriever(
    driver,
    index_name="chunkVector",
    embedder=embedding_provider,
    return_properties=["id", "text"],
)
rag = GraphRAG(retriever=vec_retriever, llm=llm)
query_text = "What is GPT?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)

# VectorCypherRetriever
retrieval_query = """
MATCH
(node)-[:HAS_ENTITY]->(t:Technology)
RETURN
node.id AS chunk_id,
node.text AS chunk_text,
collect(t.id) AS tech;
"""
vec_cyher_retriever = VectorCypherRetriever(
    driver,
    index_name="chunkVector",
    embedder=embedding_provider,
    retrieval_query=retrieval_query,
)
rag = GraphRAG(retriever=vec_cyher_retriever, llm=llm)
query_text = "What technology is used for GPT?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)

# HybridRetriever
hybrid_retriever = HybridRetriever(
    driver=driver,
    vector_index_name="chunkVector",
    fulltext_index_name="chunkText",
    embedder=embedding_provider,
    return_properties=["id", "text"],
)
rag = GraphRAG(retriever=hybrid_retriever, llm=llm)
query_text = "What is LLM and how it is applied?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)

# HybridCypherRetriever
retrieval_query = """
MATCH
(node)-[:HAS_ENTITY]->(e:Event)
RETURN
node.id AS chunk_id,
node.text AS chunk_text,
collect(e.id) AS event;
"""
hybrid_cypher_retriever = HybridCypherRetriever(
    driver=driver,
    vector_index_name="chunkVector",
    fulltext_index_name="chunkText",
    retrieval_query=retrieval_query,
    embedder=embedding_provider,
)
rag = GraphRAG(retriever=hybrid_cypher_retriever, llm=llm)
query_text = "What are the events about LLM? "
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)

# Text2CypherRetriever
neo4j_schema="""
Node with properties:
Chunk {id: STRING, text: STRING, textEmbedding: LIST}
Concept {id: STRING, provider: STRING}
Document {id: STRING}
Event {id: STRING}
Object {id: STRING}
Person {id: STRING}
Technology {id: STRING}

The relationships:
(:Chunk)-[:PART_OF]->(:Document)
(:Chunk)-[:HAS_ENTITY]->(:Object)
(:Chunk)-[:HAS_ENTITY]->(:Event)
(:Chunk)-[:HAS_ENTITY]->(:Technology)
(:Chunk)-[:HAS_ENTITY]->(:Person)
(:Chunk)-[:HAS_ENTITY]->(:Concept)
(:Concept)-[:HAS]->(:Concept)
(:Concept)-[:USES]->(:Concept)
"""
tx2cypher_retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=neo4j_schema
)
rag = GraphRAG(retriever=tx2cypher_retriever, llm=llm)
query_text = "What does 'Tom Hanks' do for the LLM study?"
response = rag.search(query_text=query_text)
print(response)