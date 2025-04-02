import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from custom_ollama_llm import OllamaLLM

load_dotenv()

llm = OllamaLLM(
    base_url=os.getenv('OLLAMA_SERVER'),
    model=os.getenv('LLM_MODEL'),
    temperature=0.0
)

embedding_provider = OllamaEmbeddings(
    base_url=os.getenv('OLLAMA_SERVER'),
    model=os.getenv('EMBEDDING_MODEL')
)