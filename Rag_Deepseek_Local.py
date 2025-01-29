import os
import base64
import gc
import random
import tempfile
import time
import uuid
from typing import Any, List, Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import Field, PrivateAttr
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import streamlit as st
import streamlit.components.v1 as components

from IPython.display import Markdown, display
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

# Configure page settings
st.set_page_config(
    page_title="DeepSeek RAG",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://deepseek.com',
        'About': "Powered by DeepSeek LLM and LlamaIndex"
    }
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #F8F9FA;
        border-radius: 20px;
        padding: 2rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2B303C 0%, #1A1D24 100%);
        color: white;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* User message */
    [data-testid="stChatMessageUser"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
    }
    
    /* Assistant message */
    [data-testid="stChatMessageAssistant"] {
        background: #F3F4F6;
        border: 1px solid #E5E7EB;
    }
    
    /* PDF preview header */
    .pdf-preview h3 {
        color: white !important;
        border-bottom: 2px solid #4F46E5;
        padding-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton button {
        background: #4F46E5;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: #4338CA;
        transform: translateY(-1px);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4F46E5;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(79, 70, 229, 0.05);
    }
    
    /* Progress indicators */
    .stProgress > div > div > div {
        background-color: #4F46E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "deepseek-rag"

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

class DeepSeekLLM(LLM):
    api_key: Optional[str] = Field(default=None, description="DeepSeek API key")
    model: str = Field(default="deepseek-reasoner", description="Model name to use")
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    max_tokens: int = Field(default=4000, description="Maximum number of tokens to generate")
    
    _client: Any = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
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
        base_client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self._client = wrap_openai(base_client)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=self.max_tokens,
            model_name=self.model,
        )

    @traceable(name="deepseek_complete")
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )
        
        return CompletionResponse(text=response.choices[0].message.content)

    @traceable(name="deepseek_stream_complete")
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        
        def gen() -> CompletionResponseGen:
            text = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    text += chunk.choices[0].delta.content
                    yield CompletionResponse(text=text, delta=chunk.choices[0].delta.content)
        
        return gen()

    @traceable(name="deepseek_chat")
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponse:
        formatted_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )
        
        return CompletionResponse(text=response.choices[0].message.content)

    @traceable(name="deepseek_stream_chat")
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
        formatted_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        
        def gen() -> CompletionResponseGen:
            text = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    text += chunk.choices[0].delta.content
                    yield CompletionResponse(text=text, delta=chunk.choices[0].delta.content)
        
        return gen()

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
def load_llm():
    return DeepSeekLLM()

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
            DeepSeek RAG
        </span>
    </h1>
    <p style="color: #6B7280; font-size: 1.1rem;">
        Intelligent Document Understanding with AI
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
                            if os.path.exists(temp_dir):
                                loader = SimpleDirectoryReader(
                                    input_dir=temp_dir,
                                    required_exts=[".pdf"],
                                    recursive=True
                                )
                            else:
                                st.error('Could not find the file you uploaded, please check again...')
                                st.stop()

                            st.write("Loading document...")
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
                            @traceable(name="create_index", run_type="embedding")
                            def create_index(documents):
                                return VectorStoreIndex.from_documents(
                                    documents,
                                    show_progress=True
                                )
                            
                            index = create_index(docs)
                            return index

                        index = process_document()

                        st.write("Setting up query engine...")
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
                        
                        query_engine = index.as_query_engine(
                            streaming=True,
                            similarity_top_k=3,
                            text_qa_template=qa_template
                        )

                        st.write("Caching results...")
                        st.session_state.file_cache[file_key] = query_engine
                    except Exception as e:
                        st.error(f"Error during document processing: {str(e)}")
                        import traceback
                        st.error(f"Detailed error: {traceback.format_exc()}")
                        st.stop()
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Outer error: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            st.stop()

# Chat interface
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("### 💬 Chat Interface")
    st.caption("Ask questions about your document and get AI-powered insights")

with col2:
    st.button("🧹 Clear Chat", on_click=reset_chat, use_container_width=True)

# Add interactive button effects
components.html("""
<script>
const streamlitDoc = window.parent.document;
const buttons = Array.from(streamlitDoc.querySelectorAll('.stButton button'));
buttons.forEach(button => {
    button.style.fontWeight = '600';
    button.style.transition = 'all 0.3s ease';
});
</script>
""")

# Chat history handling
if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat processing
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            streaming_response = query_engine.query(prompt)
            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error during query processing: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": full_response})