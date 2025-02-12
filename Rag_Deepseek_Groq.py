import os
import base64
import gc
import random
import tempfile
import time
import uuid
from typing import Any, List, Optional, Dict
from groq import Groq
from dotenv import load_dotenv
from pydantic import Field, PrivateAttr
from langsmith import traceable
import streamlit as st
import streamlit.components.v1 as components

from llama_index.core import Settings
from llama_index.core.llms import (
    ChatMessage,
    CompletionResponse,
    LLMMetadata,
    CompletionResponseGen,
    LLM,
)
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext

# Configure page settings
st.set_page_config(
    page_title="DeepSeek RAG (Groq)",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://groq.com',
        'About': "Powered by DeepSeek on Groq"
    }
)

# Load environment variables
load_dotenv()

class GroqDeepSeekLLM(LLM):
    api_key: Optional[str] = Field(default=None, description="Groq API key")
    model: str = Field(default="deepseek-r1-distill-llama-70b", description="Model name to use")
    temperature: float = Field(default=0.6, description="Temperature for sampling")
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens to generate")
    
    _client: Any = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-r1-distill-llama-70b",
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        self._client = Groq(api_key=self.api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128000,  # 128k context window
            num_output=self.max_tokens or 4096,
            model_name=self.model,
        )

    def _convert_messages_to_chat_format(self, messages: List[Dict]) -> List[Dict]:
        return [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in messages
        ]

    @traceable(name="groq_complete")
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            messages = [{"role": "user", "content": prompt}]
            chat_messages = self._convert_messages_to_chat_format(messages)
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            return CompletionResponse(text=response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error during completion: {str(e)}")
            return CompletionResponse(text="I apologize, but I encountered an error. Please try again.")

    @traceable(name="groq_stream_complete")
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            messages = [{"role": "user", "content": prompt}]
            chat_messages = self._convert_messages_to_chat_format(messages)
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            def gen() -> CompletionResponseGen:
                text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        text += chunk.choices[0].delta.content
                        yield CompletionResponse(text=text, delta=chunk.choices[0].delta.content)
            
            return gen()
        except Exception as e:
            st.error(f"Error during streaming: {str(e)}")
            def error_gen():
                yield CompletionResponse(
                    text="I apologize, but I encountered an error. Please try again.",
                    delta="I apologize, but I encountered an error. Please try again."
                )
            return error_gen()

    @traceable(name="groq_chat")
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponse:
        try:
            chat_messages = [
                {
                    "role": "system" if msg.role.value == "system" else msg.role.value,
                    "content": msg.content
                }
                for msg in messages
            ]
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            return CompletionResponse(text=response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error during chat: {str(e)}")
            return CompletionResponse(text="I apologize, but I encountered an error. Please try again.")

    @traceable(name="groq_stream_chat")
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
        try:
            chat_messages = [
                {
                    "role": "system" if msg.role.value == "system" else msg.role.value,
                    "content": msg.content
                }
                for msg in messages
            ]
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            def gen() -> CompletionResponseGen:
                text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        text += chunk.choices[0].delta.content
                        yield CompletionResponse(text=text, delta=chunk.choices[0].delta.content)
            
            return gen()
        except Exception as e:
            st.error(f"Error during streaming chat: {str(e)}")
            def error_gen():
                yield CompletionResponse(
                    text="I apologize, but I encountered an error. Please try again.",
                    delta="I apologize, but I encountered an error. Please try again."
                )
            return error_gen()

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError("Async methods are not implemented")

    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Async methods are not implemented")

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError("Async methods are not implemented")

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Async methods are not implemented")

class TracedHuggingFaceEmbedding(HuggingFaceEmbedding):
    @traceable(name="embed_documents", run_type="embedding")
    def _get_text_embedding(self, text: str) -> List[float]:
        return super()._get_text_embedding(text)
    
    @traceable(name="embed_batch", run_type="embedding")
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return super()._get_text_embeddings(texts)

@st.cache_resource
def initialize_existing_index():
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True,
            https=True
        )
        
        # Check if collection exists
        client.get_collection("deepseek_rag_docs")
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="deepseek_rag_docs"
        )
        
        return VectorStoreIndex.from_vector_store(vector_store)
    
    except Exception:
        return None

@st.cache_resource
def load_llm():
    Settings.embed_model = TracedHuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        trust_remote_code=True
    )
    Settings.llm = GroqDeepSeekLLM()
    return Settings.llm

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []

session_id = st.session_state.id
llm = load_llm()

# Initialize existing index if available
if "query_engine" not in st.session_state:
    existing_index = initialize_existing_index()
    if existing_index:
        st.session_state.query_engine = existing_index.as_query_engine(
            streaming=True,
            similarity_top_k=3,
            text_qa_template=PromptTemplate(
                "You are a helpful assistant that provides accurate answers based on the given context.\n"
                "Context information is below:\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given this context, answer the question: {query_str}\n"
                "If the answer isn't in the context, say 'I couldn't find this information in the document.' "
                "Otherwise, provide a detailed answer. Answer:"
            )
        )
        st.session_state.file_cache = {"preloaded": st.session_state.query_engine}

@traceable(name="reset_chat")
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

@traceable(name="display_pdf")
def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

# Modern header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="font-size: 2.5rem; color: #1F2937; margin-bottom: 0.5rem;">
        <span style="background: linear-gradient(45deg, #4F46E5, #9333EA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            DeepSeek RAG (Groq)
        </span>
    </h1>
    <p style="color: #6B7280; font-size: 1.1rem;">
        Ultra-fast Document Understanding with Groq
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem; border-radius: 15px; background: rgba(255,255,255,0.05);">
        <h2 style="color: white; margin-bottom: 1.5rem;">📄 Document Management</h2>
        <div style="margin-bottom: 2rem;">
            <p style="color: #9CA3AF; font-size: 0.9rem; margin-bottom: 1rem;">
                Upload PDF documents to start intelligent analysis
            </p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose PDF File",
        type="pdf",
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")
                    st.stop()

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    try:
                        @traceable(name="process_document")
                        def process_document():
                            loader = SimpleDirectoryReader(
                                input_dir=temp_dir,
                                required_exts=[".pdf"],
                                recursive=True
                            )
                            docs = loader.load_data()
                            
                            st.write("Setting up embedding model...")
                            embed_model = TracedHuggingFaceEmbedding(
                                model_name="BAAI/bge-large-en-v1.5",
                                trust_remote_code=True
                            )
                            
                            st.write("Setting up LLM...")
                            llm = load_llm()
                            
                            st.write("Configuring settings...")
                            Settings.embed_model = embed_model
                            Settings.llm = llm
                            Settings.chunk_size = 512
                            Settings.chunk_overlap = 50
                            
                            st.write("Creating document index...")
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
                            
                            storage_context = StorageContext.from_defaults(vector_store=vector_store)
                            index = VectorStoreIndex.from_documents(
                                docs,
                                storage_context=storage_context,
                                show_progress=True
                            )
                            
                            return index

                        index = process_document()
                        
                        st.write("Setting up query engine...")
                        query_engine = index.as_query_engine(
                            streaming=True,
                            similarity_top_k=3,
                            text_qa_template=PromptTemplate(
                                "You are a helpful assistant that provides accurate answers based on the given context.\n"
                                "Context information is below:\n"
                                "---------------------\n"
                                "{context_str}\n"
                                "---------------------\n"
                                "Given this context, answer the question: {query_str}\n"
                                "If the answer isn't in the context, say 'I couldn't find this information in the document.' "
                                "Otherwise, provide a detailed answer. Answer:"
                            )
                        )

                        st.write("Caching results...")
                        st.session_state.file_cache[file_key] = query_engine
                        st.session_state.query_engine = query_engine
                    except Exception as e:
                        st.error(f"Error during document processing: {str(e)}")
                        st.stop()
                else:
                    query_engine = st.session_state.file_cache[file_key]
                    st.session_state.query_engine = query_engine

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

# Chat interface
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("### 💬 Chat Interface")
    st.caption("Ask questions about your document and get ultra-fast AI-powered insights")

with col2:
    st.button("🧹 Clear Chat", on_click=reset_chat, use_container_width=True)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if 'query_engine' not in st.session_state:
                st.error("Please upload a PDF document first!")
                st.stop()
                
            streaming_response = st.session_state.query_engine.query(prompt)
            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error during query processing: {str(e)}")
            st.stop()

        st.session_state.messages.append({"role": "assistant", "content": full_response}) 