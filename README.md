# DeepseekRAG Documentation

## Overview

The DeepseekRAG implementation is a sophisticated Retrieval-Augmented Generation (RAG) system that combines the power of DeepSeek's language model with document retrieval capabilities. Built using LlamaIndex and Streamlit, it provides an interactive interface for users to chat with their PDF documents using state-of-the-art language models.

## Required Libraries

### Core Dependencies
```txt
# requirements.txt
llama-index-core==0.12.12
llama-index-embeddings-huggingface==0.5.1
streamlit==1.41.1
python-dotenv==1.0.1
openai==1.59.9
transformers==4.48.1
sentence-transformers==3.3.1
pypdf==5.1.0
pydantic==2.10.5
```

### Library Descriptions

1. **LlamaIndex Core** (v0.12.12)
   - Purpose: Core framework for building RAG applications
   - Installation: `pip install llama-index-core`
   - Features: Document indexing, query engine, service context

2. **LlamaIndex HuggingFace Embeddings** (v0.5.1)
   - Purpose: Integration with HuggingFace embedding models
   - Installation: `pip install llama-index-embeddings-huggingface`
   - Features: BAAI/bge-large-en-v1.5 embedding model support

3. **Streamlit** (v1.41.1)
   - Purpose: Web interface and UI components
   - Installation: `pip install streamlit`
   - Features: Interactive chat interface, file upload, session state

4. **OpenAI** (v1.59.9)
   - Purpose: DeepSeek API client (OpenAI-compatible)
   - Installation: `pip install openai`
   - Features: Chat completions, API integration

5. **Transformers** (v4.48.1)
   - Purpose: Transformer models and utilities
   - Installation: `pip install transformers`
   - Features: Model loading, tokenization

### Environment Setup

1. **Using Conda**:
```bash
conda create -n rag_env python=3.11.11
conda activate rag_env
pip install -r requirements.txt
```

2. **Required Environment Variables**:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### Version Compatibility
- Python: 3.11.11
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
│   PDF Upload    │────▶│   Vector DB  │◀───▶│  BAAI Embed    │
└─────────────────┘     └──────────────┘     └────────────────┘
```

## Core Components

### 1. DeepSeek LLM Integration
```python
class DeepSeekLLM(LLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.1,
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
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
    trust_remote_code=True
)
```

### 3. Document Processing
```python
loader = SimpleDirectoryReader(
    input_dir=temp_dir,
    required_exts=[".pdf"],
    recursive=True
)
docs = loader.load_data()
```

### 4. Index Creation
```python
index = VectorStoreIndex.from_documents(
    docs,
    show_progress=True
)
```

## Features

1. **PDF Document Support**
   - Upload and process PDF files
   - Automatic text extraction
   - Document chunking and indexing

2. **Advanced RAG Capabilities**
   - Semantic search using BAAI embeddings
   - Context-aware responses
   - Streaming response generation

3. **User Interface**
   - Interactive chat interface
   - PDF preview functionality
   - Progress indicators
   - Error handling and feedback

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
    similarity_top_k=50,
    text_qa_template=qa_template
)
```

## Error Handling

1. **File Processing Errors**
```python
try:
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
except Exception as e:
    st.error(f"Error saving file: {str(e)}")
    st.stop()
```

2. **Query Processing Errors**
```python
try:
    response = query_engine.query(prompt)
except Exception as e:
    st.error(f"Error during query processing: {str(e)}")
```

## Performance Optimization

1. **Memory Management**
   - Session state cleanup
   - Garbage collection
   - File caching

2. **Embedding Optimization**
   - Chunk size: 512 tokens
   - Chunk overlap: 50 tokens
   - Top-k retrieval: 50 documents

3. **Response Generation**
   - Streaming responses
   - Temperature: 0.1 (focused responses)
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

## Usage Guide

1. **Starting the Application**
```bash
streamlit run Rag_Deepseek_Local.py
```

2. **Document Upload**
   - Click "Choose your `.pdf` file"
   - Wait for indexing completion
   - Start chatting with your document

3. **Interacting with Documents**
   - Ask questions about the content
   - View source context
   - Get real-time responses

## Maintenance

1. **Dependencies**
   - Regular updates of core packages
   - Compatibility checks
   - Security patches

2. **Monitoring**
   - Memory usage
   - API response times
   - Error logging

This documentation provides a comprehensive guide to understanding and implementing the DeepseekRAG system. The implementation combines state-of-the-art language models with efficient document retrieval for an enhanced question-answering experience. 