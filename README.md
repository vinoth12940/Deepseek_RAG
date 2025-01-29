# DeepseekRAG Documentation

## Overview

The DeepseekRAG implementation is a sophisticated Retrieval-Augmented Generation (RAG) system that combines the power of DeepSeek's language model with document retrieval capabilities. Built using LlamaIndex, Streamlit, and Qdrant vector store, it provides an interactive interface for users to chat with their PDF documents using state-of-the-art language models.

## Required Libraries

### Core Dependencies
```txt
streamlit
python-dotenv
openai
langsmith
llama-index
ipython
pydantic
torch
transformers
sentence-transformers
pypdf
qdrant-client
llama-index-vector-stores-qdrant
```

### Library Descriptions

1. **LlamaIndex**
   - Purpose: Core framework for building RAG applications
   - Features: Document indexing, query engine, service context
   - Includes vector store integration with Qdrant

2. **LangSmith**
   - Purpose: Tracing and monitoring for LLM applications
   - Features: Performance tracking, debugging, and optimization

3. **Streamlit**
   - Purpose: Modern web interface and UI components
   - Features: Interactive chat interface, file upload, session state
   - Enhanced UI with custom CSS styling

4. **OpenAI**
   - Purpose: DeepSeek API client (OpenAI-compatible)
   - Features: Chat completions, streaming responses

5. **Qdrant**
   - Purpose: Vector database for document storage
   - Features: Cloud-hosted vector storage, efficient similarity search

### Environment Setup

1. **Using pip**:
```bash
pip install -r requirements.txt
```

2. **Required Environment Variables**:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_instance_url
LANGCHAIN_API_KEY=your_langsmith_api_key
```

### Version Compatibility
- Python: 3.11+
- OS Support: Windows, macOS, Linux
- CUDA: Optional (for GPU acceleration)

## System Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  Streamlit UI   │────▶│  LlamaIndex  │────▶│  DeepSeek LLM  │
└─────────────────┘     └──────────────┘     └────────────────┘
        │                      │                      │
        │                      │                      │
        ▼                      ▼                      ▼
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│   PDF Upload    │────▶│ Qdrant Store │◀───▶│  BAAI Embed    │
└─────────────────┘     └──────────────┘     └────────────────┘
```

## Core Components

### 1. DeepSeek LLM Integration
```python
class DeepSeekLLM(LLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-reasoner",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
```

### 2. Embedding Configuration
```python
embed_model = TracedHuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
    trust_remote_code=True
)
```

### 3. Vector Store Setup
```python
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=True,
    https=True
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="deepseek_rag_docs"
)
```

## Features

1. **Enhanced Document Processing**
   - Cloud-based vector storage with Qdrant
   - Automatic text extraction and chunking
   - Persistent document indexing

2. **Advanced RAG Capabilities**
   - Semantic search using BAAI embeddings
   - Context-aware responses
   - Streaming response generation
   - Performance tracing with LangSmith

3. **Modern User Interface**
   - Beautiful gradient-styled components
   - Interactive chat interface
   - PDF preview functionality
   - Real-time progress indicators
   - Comprehensive error handling

## Query Processing

### 1. Custom QA Template
```python
qa_template = PromptTemplate(
    "You are a helpful assistant that provides accurate answers based on the given context. "
    "Context information is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this context, please answer the question: {query_str}\n\n"
    "If the answer is not contained in the context, say 'I cannot find this information in the provided context.' "
    "Answer:"
)
```

### 2. Query Engine Configuration
```python
query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=3,
    text_qa_template=qa_template
)
```

## Performance Optimization

1. **Memory Management**
   - Session state cleanup
   - Garbage collection
   - File caching
   - Cloud-based vector storage

2. **Embedding Optimization**
   - Chunk size: 512 tokens
   - Chunk overlap: 50 tokens
   - Top-k retrieval: 3 documents

3. **Response Generation**
   - Streaming responses
   - Temperature: 0.3 (balanced responses)
   - Context window: 8192 tokens

## Security Considerations

1. **File Handling**
   - Temporary file storage
   - File type validation
   - Size limitations

2. **API Security**
   - Secure key storage
   - Environment variables
   - Session isolation
   - HTTPS-enabled Qdrant connection

## Usage Guide

1. **Starting the Application**
```bash
streamlit run Rag_Deepseek_Local.py
```

2. **Document Upload**
   - Click "Choose PDF File"
   - Wait for cloud indexing completion
   - Start chatting with your document

3. **Interacting with Documents**
   - Ask questions about the content
   - View source context
   - Get real-time responses
   - Monitor performance with LangSmith

## Maintenance

1. **Dependencies**
   - Regular updates of core packages
   - Compatibility checks
   - Security patches

2. **Monitoring**
   - LangSmith tracing
   - Qdrant health checks
   - Memory usage
   - API response times

This documentation provides a comprehensive guide to understanding and implementing the DeepseekRAG system. The implementation combines state-of-the-art language models with efficient cloud-based document retrieval for an enhanced question-answering experience. 