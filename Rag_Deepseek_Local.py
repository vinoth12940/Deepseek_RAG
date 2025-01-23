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

import streamlit as st

# Load environment variables
load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "deepseek-rag"  # You can change this project name

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
        model: str = "deepseek-reasoner",
        temperature: float = 0.5,
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
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.header(f"Add your documents!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

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
                        # Custom QA prompt
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
                            similarity_top_k=50,  # Retrieve more context
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

                # Inform the user that the file is processed and Display the PDF uploaded
                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Outer error: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            st.stop()

col1, col2 = st.columns([6, 1])

# st.markdown("""
#     # Agentic RAG powered by <img src="data:image/png;base64,{}" width="120" style="vertical-align: -3px;">
# """.format(base64.b64encode(open("assets/crewai.png", "rb").read()).decode()), unsafe_allow_html=True)


with col1:
    # st.header(f"Chat with Docs using Llama-3.3")
    st.markdown("""
    # RAG powered by <img src="data:image/png;base64,{}" width="150" style="vertical-align: -3px;">
""".format(base64.b64encode(open("assets/DEEPSEEK.png", "rb").read()).decode()), unsafe_allow_html=True)

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            @traceable(name="process_query")
            def process_query(query_text):
                response = query_engine.query(query_text)
                if hasattr(response, 'response'):
                    return response.response
                return str(response)
            
            full_response = process_query(prompt)
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Error during query processing: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            st.stop()

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})