from typing import Optional, Dict, Any, List, Union, Callable
from langchain_ollama import ChatOllama
from neo4j_graphrag.llm import OpenAILLM

class OllamaLLM(OpenAILLM):
    """Custom LLM implementation for Ollama models that extends the OpenAILLM class."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        **kwargs
    ):
        # Initialize with the real OpenAI key if available, otherwise dummy
        import os
        api_key = os.getenv("OPENAI_API_KEY", "sk-dummy-key-for-validation")
        super().__init__(api_key=api_key, model_name="gpt-3.5-turbo")
        
        # Store our actual parameters
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.ollama_chat = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature
        )
        
        # Override the OpenAI client to prevent API calls
        self._override_client()
        
    def _override_client(self):
        """Override the OpenAI client to match the expected structure."""
        class Completions:
            def create(self, messages, model=None, **kwargs):
                # Convert messages to LangChain format
                from langchain.schema import HumanMessage, AIMessage, SystemMessage
                
                lc_messages = []
                for message in messages:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    
                    if role == "user":
                        lc_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                    elif role == "system":
                        lc_messages.append(SystemMessage(content=content))
                
                # Use the Ollama chat model to get a response
                ollama_llm = OllamaLLM.instance.ollama_chat
                response = ollama_llm.invoke(lc_messages)
                
                # Return in the format expected by OpenAILLM
                return type('obj', (object,), {
                    'choices': [
                        type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': response.content
                            })
                        })
                    ]
                })

        class Chat:
            def __init__(self):
                self.completions = Completions()

        class Embeddings:
            def create(self, input, model=None, **kwargs):
                # Use the Ollama model for embeddings
                from langchain_community.embeddings import OllamaEmbeddings
                
                embedder = OllamaEmbeddings(
                    base_url=OllamaLLM.instance.base_url,
                    model=OllamaLLM.instance.model
                )
                
                if isinstance(input, str):
                    embedding = embedder.embed_query(input)
                else:
                    embedding = embedder.embed_documents(input)
                
                # Return in the format expected by OpenAILLM
                return type('obj', (object,), {
                    'data': [
                        type('obj', (object,), {
                            'embedding': embedding if isinstance(embedding, list) else [embedding]
                        })
                    ]
                })

        class Models:
            def list(self):
                return {"data": [{"id": "gpt-3.5-turbo"}]}

        # Create a new client with the right structure
        self.client = type('obj', (object,), {
            'chat': Chat(),
            'embeddings': Embeddings(),
            'models': Models()
        })
        
        # Save instance for use in the completions create method
        OllamaLLM.instance = self

    # We don't need to override invoke since we're providing a compatible client structure
    # that the parent's invoke method can use