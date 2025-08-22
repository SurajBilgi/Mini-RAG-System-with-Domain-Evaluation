import streamlit as st
import os
import tempfile
import time
from typing import List, Dict
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Error handling for imports
try:
    # LangChain imports
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import Document
    from langchain.callbacks.base import BaseCallbackHandler

    # Document loaders
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        CSVLoader,
        UnstructuredWordDocumentLoader,
    )

    # Evaluation imports
    import textstat
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Hugging Face imports
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    from langchain.chains import RetrievalQA
    from transformers import pipeline
    import torch

except ImportError as e:
    st.error(f"Import error: {e}")
    st.error(
        "Please install the required dependencies with: "
        "pip install -r requirements.txt"
    )
    st.stop()


# Enhanced CSS for beautiful and modern UI
def load_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, 
                                   #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease-out;
    }
    
    .subheader {
        font-size: 1.3rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        animation: pulse 2s infinite;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .chat-message {
        padding: 1.2rem 1.5rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        position: relative;
        animation: slideInLeft 0.5s ease-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .user-message {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        margin-left: 15%;
        border-bottom-right-radius: 5px;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        color: #374151;
        margin-right: 15%;
        border: 1px solid #e5e7eb;
        border-bottom-left-radius: 5px;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .evaluation-toggle {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 1px solid #a7f3d0;
        position: relative;
        overflow: hidden;
    }
    
    .evaluation-toggle::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #10b981 0%, #059669 100%);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 0.5rem;
        border-radius: 20px;
        border: 2px dashed #f59e0b;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%);
        border-color: #ea580c;
        transform: scale(1.02);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    .sample-questions {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid #a5b4fc;
    }
    
    .source-doc {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #7dd3fc;
        transition: all 0.3s ease;
    }
    
    .source-doc:hover {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        transform: translateX(5px);
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 4px;
        border-radius: 2px;
        animation: shimmer 2s infinite linear;
    }
    
    .floating-action {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        cursor: pointer;
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .floating-action:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: 200px 0; }
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 1px solid #d1d5db;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #d1d5db;
        padding: 0.75rem;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 1px solid #d1d5db;
        padding: 0.75rem;
    }
    
    .stFileUploader > div {
        border-radius: 15px;
        border: 2px dashed #9ca3af;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background-color: #f8fafc;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            color: #f9fafb;
            border-color: #374151;
        }
        
        .chat-container {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border-color: #374151;
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
            color: #f9fafb;
            border-color: #4b5563;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(
            f'<div class="assistant-message">'
            f"<strong>🤖 Assistant:</strong><br>{self.text}</div>",
            unsafe_allow_html=True,
        )


class RAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.chain = None
        self.memory = None
        self.documents = []
        self.model_provider = "openai"  # Track which provider is being used
        self.hf_pipeline = None  # Store HF pipeline

    def initialize_embeddings(
        self, api_key: str, model_name: str = "text-embedding-ada-002"
    ):
        """Initialize OpenAI embeddings"""
        try:
            os.environ["OPENAI_API_KEY"] = api_key
            self.embeddings = OpenAIEmbeddings(model=model_name)
            self.model_provider = "openai"
            print(f"✅ Embeddings initialized with {model_name}")
        except Exception as e:
            raise ValueError(f"Failed to initialize embeddings: {str(e)}")

    def initialize_hf_embeddings(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize Hugging Face embeddings"""
        try:
            model_kwargs = {"device": "cpu"}
            if torch.cuda.is_available():
                model_kwargs["device"] = "cuda"
                print("🚀 Using GPU acceleration")

            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
            self.model_provider = "huggingface"
            print(f"✅ Hugging Face Embeddings initialized with {model_name}")
        except Exception as e:
            raise ValueError(f"Failed to initialize HF embeddings: {str(e)}")

    def load_documents(self, uploaded_files, text_input: str = "") -> List[Document]:
        """Load documents from various sources"""
        documents = []

        # Process uploaded files
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.name.endswith(".txt"):
                    # Use TextLoader with UTF-8 encoding
                    loader = TextLoader(tmp_file_path, encoding="utf-8")
                elif uploaded_file.name.endswith(".csv"):
                    loader = CSVLoader(tmp_file_path)
                elif uploaded_file.name.endswith((".doc", ".docx")):
                    loader = UnstructuredWordDocumentLoader(tmp_file_path)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue

                docs = loader.load()
                if not docs:
                    st.warning(f"No content found in {uploaded_file.name}")
                    continue

                for doc in docs:
                    if doc.page_content and doc.page_content.strip():
                        # Clean and encode document content properly
                        doc.page_content = self._clean_text(doc.page_content)
                        doc.metadata["source"] = uploaded_file.name
                        documents.append(doc)

            except Exception as e:
                # Use safe error message without Unicode characters
                error_msg = str(e).encode("ascii", "ignore").decode("ascii")
                st.error(f"Error loading {uploaded_file.name}: {error_msg}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        # Process text input
        if text_input.strip():
            # Clean the text input to handle Unicode properly
            clean_text = self._clean_text(text_input)
            if clean_text.strip():
                documents.append(
                    Document(
                        page_content=clean_text,
                        metadata={
                            "source": "Text Input",
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                )

        print(f"📚 Loaded {len(documents)} documents")
        return documents

    def _clean_text(self, text: str) -> str:
        """Clean text to handle Unicode and encoding issues"""
        if not text:
            return ""

        try:
            # Remove or replace problematic Unicode characters
            # Keep most Unicode but replace problematic ones
            cleaned = text.replace("\u274c", "X")  # Replace ❌ with X
            cleaned = cleaned.replace("\u2705", "OK")  # Replace ✅ with OK
            cleaned = cleaned.replace("\u26a0", "!")  # Replace ⚠️ with !

            # Ensure the text can be encoded as UTF-8
            cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8")

            # Remove any remaining control characters except common ones
            import re

            cleaned = re.sub(
                r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]", "", cleaned
            )

            return cleaned.strip()
        except Exception:
            # Fallback: return ASCII-safe version
            return text.encode("ascii", "ignore").decode("ascii").strip()

    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """Split documents into chunks with better overlap for retrieval"""
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Better separators
        )

        chunks = text_splitter.split_documents(documents)

        # Add chunk information to metadata for better tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

    def create_vectorstore(self, chunks: List[Document]):
        """Create vector store from document chunks with validation"""
        if not chunks:
            raise ValueError("No documents to process")

        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Please set up API key first.")

        # Validate that chunks have content
        valid_chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content and len(chunk.page_content.strip()) > 10
        ]

        if not valid_chunks:
            raise ValueError(
                "No valid content found in documents. Please check your files."
            )

        print(f"🔍 Processing {len(valid_chunks)} valid chunks...")

        try:
            # Use in-memory ChromaDB to avoid tenant/connection issues
            print("📦 Creating vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=valid_chunks,
                embedding=self.embeddings,
                # No persist_directory = in-memory storage
            )

            # Test the vector store
            print("🧪 Testing vector store...")
            test_results = self.vectorstore.similarity_search("test query", k=1)

            # Log for debugging
            print(f"✅ Created in-memory vector store with {len(valid_chunks)} chunks")
            print(f"✅ Vector store test: {len(test_results)} results")
            if test_results:
                print(f"📝 Test result sample: {test_results[0].page_content[:100]}...")

        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            # Fallback: Try with explicit collection name
            try:
                print("🔄 Trying fallback method...")
                self.vectorstore = Chroma.from_documents(
                    documents=valid_chunks,
                    embedding=self.embeddings,
                    collection_name="rag_documents",
                )
                print(
                    "✅ Successfully created vector store with explicit collection name"
                )
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                raise ValueError(f"Failed to create vector store: {str(e)}")

        print("🎯 Vector store ready for queries!")

    def setup_qa_chain(
        self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1
    ):
        """Setup the conversational QA chain"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        # Initialize LLM based on provider
        if self.model_provider == "openai":
            llm = ChatOpenAI(model=model_name, temperature=temperature, streaming=True)

            # Set up memory for OpenAI (conversational)
            self.memory = ConversationBufferWindowMemory(
                k=10,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
        elif self.model_provider == "huggingface":
            # Initialize Hugging Face pipeline
            if not self.hf_pipeline:
                try:
                    # Use a simple, reliable model for text generation
                    valid_hf_models = ["gpt2", "distilgpt2", "microsoft/DialoGPT-small"]
                    hf_model = (
                        model_name if model_name in valid_hf_models else "distilgpt2"
                    )

                    print(f"Loading Hugging Face model: {hf_model}")

                    # Try a simpler approach with direct model loading
                    from transformers import AutoTokenizer, AutoModelForCausalLM

                    tokenizer = AutoTokenizer.from_pretrained(hf_model)
                    model = AutoModelForCausalLM.from_pretrained(hf_model)

                    # Set pad token if not present
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                    # Create a custom LLM class
                    class SimpleHuggingFaceLLM(HuggingFacePipeline):
                        def __init__(self, tokenizer, model):
                            self.tokenizer = tokenizer
                            self.model = model

                        def _call(self, prompt: str, stop=None, run_manager=None):
                            try:
                                # Prepare input with better formatting
                                formatted_prompt = (
                                    f"Context: {prompt}\n\nProvide a helpful answer:\n"
                                )

                                # Tokenize input
                                inputs = self.tokenizer.encode(
                                    formatted_prompt, return_tensors="pt"
                                )

                                # Limit input length
                                if inputs.shape[1] > 400:
                                    inputs = inputs[:, -400:]  # Take last 400 tokens

                                # Generate with proper parameters
                                with torch.no_grad():
                                    outputs = self.model.generate(
                                        inputs,
                                        max_new_tokens=50,
                                        min_length=inputs.shape[1] + 10,
                                        do_sample=True,
                                        temperature=0.8,
                                        top_p=0.9,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        repetition_penalty=1.2,
                                        num_return_sequences=1,
                                    )

                                # Decode output
                                response = self.tokenizer.decode(
                                    outputs[0], skip_special_tokens=True
                                )

                                # Clean response
                                if formatted_prompt in response:
                                    response = response.replace(
                                        formatted_prompt, ""
                                    ).strip()

                                print(f"Raw model output: '{response}'")

                                # Further cleaning
                                if response and len(response.strip()) > 0:
                                    # Remove common artifacts
                                    cleaned = response.strip()
                                    lines = [
                                        line.strip()
                                        for line in cleaned.split("\n")
                                        if line.strip()
                                    ]
                                    if lines:
                                        cleaned = " ".join(
                                            lines[:3]
                                        )  # Take first 3 non-empty lines

                                    print(f"Cleaned response: '{cleaned}'")
                                    return (
                                        cleaned
                                        if cleaned
                                        else "I can provide information based on the documents, but need a more specific question."
                                    )

                                return "I can see the relevant information in the documents. Please try asking a more specific question."

                            except Exception as e:
                                print(f"Simple HF LLM error: {e}")
                                return "I'm having trouble generating a response. The information is available in the documents, but please try rephrasing your question."

                    self.hf_pipeline = SimpleHuggingFaceLLM(tokenizer, model)
                    print(f"✅ Simple Hugging Face model {hf_model} initialized")

                except Exception as e:
                    print(f"Error loading simple HF model: {e}")

                    # Fallback to pipeline approach
                    try:
                        # Configure pipeline
                        pipe = pipeline(
                            "text-generation",
                            model=hf_model,
                            device=-1,  # Use CPU for reliability
                            return_full_text=False,
                            max_length=None,  # Remove max_length to avoid conflict
                            truncation=True,
                        )

                        # Create a wrapper for better responses
                        class SafeHuggingFacePipeline(HuggingFacePipeline):
                            def _call(self, prompt: str, stop=None, run_manager=None):
                                try:
                                    # Clean and shorten prompt to avoid token limit issues
                                    if len(prompt) > 200:
                                        # Take the last part which usually contains the question
                                        prompt = prompt[-200:]

                                    # Simple, direct prompt format
                                    clean_prompt = f"Question: {prompt}\n\nAnswer:"

                                    result = self.pipeline(
                                        clean_prompt,
                                        max_new_tokens=100,  # Use only max_new_tokens
                                        min_length=10,  # Ensure minimum output length
                                        do_sample=True,
                                        temperature=0.7,
                                        top_p=0.9,
                                        pad_token_id=50256,
                                        eos_token_id=50256,
                                        repetition_penalty=1.2,
                                        num_return_sequences=1,
                                        truncation=True,
                                    )

                                    print(f"HF Generation result: {result}")

                                    if isinstance(result, list) and len(result) > 0:
                                        if (
                                            isinstance(result[0], dict)
                                            and "generated_text" in result[0]
                                        ):
                                            generated = result[0]["generated_text"]
                                            print(f"Raw generated text: '{generated}'")
                                            print(
                                                f"Generated text length: {len(generated)}"
                                            )
                                            if generated and len(generated.strip()) > 0:
                                                # Clean up the response
                                                cleaned = generated.strip()
                                                # Remove any repeated prompt parts
                                                if "Answer:" in cleaned:
                                                    cleaned = cleaned.split("Answer:")[
                                                        -1
                                                    ].strip()
                                                if "Question:" in cleaned:
                                                    # Remove everything before and including "Question:"
                                                    parts = cleaned.split("Question:")
                                                    cleaned = (
                                                        parts[-1].strip()
                                                        if len(parts) > 1
                                                        else cleaned
                                                    )

                                                print(f"Cleaned text: '{cleaned}'")
                                                final_answer = (
                                                    cleaned
                                                    if cleaned
                                                    else "I need more context to provide a complete answer."
                                                )
                                                print(f"Final answer: '{final_answer}'")
                                                return final_answer
                                        elif isinstance(result[0], str):
                                            print(f"String result: '{result[0]}'")
                                            return result[0].strip()

                                    return "Based on the provided documents, I can see relevant information but the model didn't generate a complete response. Please try rephrasing your question."

                                except Exception as e:
                                    print(f"HF Pipeline error: {e}")
                                    return "I'm experiencing technical difficulties with text generation. Please try a simpler question or use OpenAI."

                        self.hf_pipeline = SafeHuggingFacePipeline(pipeline=pipe)
                        print(f"✅ Fallback Hugging Face model {hf_model} initialized")

                    except Exception as e2:
                        print(f"Error loading Hugging Face model: {e2}")

                        # Simple fallback that's compatible with LangChain
                        from langchain.llms.base import LLM

                        class DummyLLM(LLM):
                            @property
                            def _llm_type(self) -> str:
                                return "dummy"

                            def _call(
                                self, prompt: str, stop=None, run_manager=None
                            ) -> str:
                                return "I'm using a fallback system due to model loading issues. Please try with OpenAI or check your Hugging Face model configuration."

                        self.hf_pipeline = DummyLLM()

            llm = self.hf_pipeline
            # No memory for Hugging Face models (simpler approach)
            self.memory = None

        # Create a custom prompt template to ensure document usage
        from langchain.prompts import PromptTemplate

        if self.model_provider == "huggingface":
            # Very simple template for HF models
            template = """Context: {context}

Question: {question}

Answer:"""

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )

            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )

            # Use RetrievalQA for Hugging Face
            self.chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
            )
        else:
            # Original OpenAI template and chain
            template = """You are a helpful AI assistant. Use the following pieces of context from the documents to answer the question. If you don't know the answer based on the provided context, say so clearly.

Context from documents:
{context}

Chat History:
{chat_history}

Question: {question}

Please provide a detailed answer based on the context from the documents above. If the context doesn't contain relevant information, clearly state that you don't have access to that specific information in the provided documents."""

            prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=template,
            )

            # Create retriever with more documents and better search
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )

            # Only use memory if it exists (OpenAI case)
            if self.memory:
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,
                    combine_docs_chain_kwargs={"prompt": prompt},
                )
            else:
                # Fallback to simple RetrievalQA if no memory
                self.chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt},
                )

    def get_response(self, query: str, callback_handler):
        """Get streaming response from the RAG system with better retrieval"""
        if not self.chain:
            raise ValueError("QA chain not initialized")

        print(f"🔍 Processing query with {self.model_provider}: {query}")

        # Test retrieval before generating response
        if self.vectorstore:
            test_docs = self.vectorstore.similarity_search(query, k=3)
            print(f"📚 Retrieved {len(test_docs)} documents for query: {query}")
            for i, doc in enumerate(test_docs):
                print(f"Doc {i+1}: {doc.page_content[:100]}...")

        try:
            # Use different input keys based on provider
            if self.model_provider == "huggingface":
                print("🤗 Using Hugging Face RetrievalQA chain")
                # RetrievalQA expects 'query' key
                response = self.chain(
                    {"query": query},
                    callbacks=[callback_handler] if callback_handler else [],
                )
                print(f"HF Response keys: {response.keys()}")
                # Extract answer and source docs from RetrievalQA response
                answer = response.get("result", "No response generated")
                source_docs = response.get("source_documents", [])
            else:
                print("🔮 Using OpenAI ConversationalRetrievalChain")
                # ConversationalRetrievalChain expects 'question' key
                response = self.chain(
                    {"question": query},
                    callbacks=[callback_handler] if callback_handler else [],
                )
                print(f"OpenAI Response keys: {response.keys()}")
                # Extract from standard ConversationalRetrievalChain response
                answer = response["answer"]
                source_docs = response.get("source_documents", [])

            print(f"✅ Final answer: {answer[:100]}...")
            print(f"📄 Source docs count: {len(source_docs)}")

            # Return unified format
            return {"answer": answer, "source_documents": source_docs}
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            import traceback

            traceback.print_exc()
            raise e


class EvaluationMetrics:
    def __init__(self):
        try:
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Error loading similarity model: {e}")
            self.similarity_model = None

    def calculate_relevance(
        self, query: str, answer: str, source_docs: List[Document]
    ) -> float:
        """Calculate relevance score using semantic similarity"""
        if not self.similarity_model:
            return 0.0

        try:
            query_embedding = self.similarity_model.encode([query])
            answer_embedding = self.similarity_model.encode([answer])

            # Calculate similarity between query and answer
            similarity = np.dot(query_embedding[0], answer_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(answer_embedding[0])
            )
            return float(similarity)
        except Exception as e:
            st.warning(f"Error calculating relevance: {e}")
            return 0.0

    def calculate_faithfulness(self, answer: str, source_docs: List[Document]) -> float:
        """Calculate faithfulness using document similarity"""
        if not self.similarity_model or not source_docs:
            return 0.0

        try:
            answer_embedding = self.similarity_model.encode([answer])
            doc_texts = [doc.page_content for doc in source_docs]
            doc_embeddings = self.similarity_model.encode(doc_texts)

            similarities = []
            for doc_embedding in doc_embeddings:
                similarity = np.dot(answer_embedding[0], doc_embedding) / (
                    np.linalg.norm(answer_embedding[0]) * np.linalg.norm(doc_embedding)
                )
                similarities.append(similarity)

            return float(np.mean(similarities)) if similarities else 0.0
        except Exception as e:
            st.warning(f"Error calculating faithfulness: {e}")
            return 0.0

    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        try:
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "automated_readability_index": textstat.automated_readability_index(
                    text
                ),
            }
        except Exception as e:
            st.warning(f"Error calculating readability: {e}")
            return {"flesch_reading_ease": 0.0}


def create_metric_card(title: str, value: str, icon: str = "📊"):
    """Create a beautiful metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def create_status_card(message: str, icon: str = "✅"):
    """Create a status card"""
    return f"""
    <div class="status-card">
        <h3>{icon} {message}</h3>
    </div>
    """


def main():
    st.set_page_config(
        page_title="🤖 Mini RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/yourusername/mini-rag-system",
            "Report a bug": "mailto:support@example.com",
            "About": "# Mini RAG System\nBuilt with ❤️ using Streamlit",
        },
    )

    load_css()

    # Enhanced Header with Animation
    st.markdown(
        '<div class="main-header">🤖 Mini RAG System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subheader">✨ Intelligent Document Analysis '
        "with Advanced AI ✨</div>",
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "evaluation_metrics" not in st.session_state:
        st.session_state.evaluation_metrics = EvaluationMetrics()
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "performance_data" not in st.session_state:
        st.session_state.performance_data = []
    if "evaluation_enabled" not in st.session_state:
        st.session_state.evaluation_enabled = False
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")

        # Model Provider Selection
        st.markdown("#### 🤖 Model Provider")
        provider = st.selectbox(
            "Choose Provider",
            ["OpenAI", "Hugging Face (Free)"],
            help="Select between paid OpenAI models or free Hugging Face models",
        )

        # Set provider key for logic
        provider_key = "openai" if provider == "OpenAI" else "huggingface"

        # API Key Section
        if provider == "OpenAI":
            st.markdown("##### 🔑 OpenAI Configuration")
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-...",
                help="Enter your OpenAI API key to get started",
            )

            if api_key:
                st.success("🔓 API Key Connected!")
            else:
                st.warning("🔒 API Key Required")
        else:  # Hugging Face
            st.markdown("##### 🤗 Hugging Face (Free)")
            st.info("🆓 No API key required! Uses local models.")
            api_key = None

        # Show configuration only if we have API key OR using Hugging Face
        if api_key or provider_key == "huggingface":
            # Model Configuration
            st.markdown("#### 🧠 AI Model Selection")

            if provider_key == "openai":
                model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
                embedding_options = [
                    "text-embedding-ada-002",
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                ]
            else:  # huggingface
                model_options = ["gpt2", "distilgpt2", "microsoft/DialoGPT-small"]
                embedding_options = [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/paraphrase-MiniLM-L6-v2",
                ]

            selected_model = st.selectbox(
                "Choose Model",
                model_options,
                help="Select the AI model for generating responses",
            )

            # Advanced Settings
            with st.expander("🔧 Advanced Settings"):
                temperature = st.slider(
                    "🌡️ Temperature",
                    0.0,
                    1.0,
                    0.1,
                    0.1,
                    help="Controls randomness in responses",
                )
                chunk_size = st.slider(
                    "📄 Chunk Size",
                    500,
                    2000,
                    1000,
                    100,
                    help="Size of document chunks for processing",
                )
                chunk_overlap = st.slider(
                    "🔄 Chunk Overlap",
                    50,
                    500,
                    200,
                    50,
                    help="Overlap between document chunks",
                )
                embedding_model = st.selectbox(
                    "🎯 Embedding Model",
                    embedding_options,
                    help="Choose the embedding model for document processing",
                )

            # Initialize embeddings based on provider
            if not st.session_state.rag_system.embeddings:
                with st.spinner("🚀 Initializing AI..."):
                    try:
                        if provider_key == "openai":
                            st.session_state.rag_system.initialize_embeddings(
                                api_key, embedding_model
                            )
                        else:  # huggingface
                            st.session_state.rag_system.initialize_hf_embeddings(
                                embedding_model
                            )
                        st.markdown(
                            create_status_card("AI System Ready!", "🚀"),
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Failed to initialize AI: {str(e)}")
                        if provider_key == "huggingface":
                            st.info(
                                "💡 First time using Hugging Face models? "
                                "They need to download first (this may take a few minutes)."
                            )

        # Enhanced Evaluation Toggle
        st.markdown("#### 📊 Evaluation Settings")

        evaluation_enabled = st.toggle(
            "🔬 Enable Detailed Analysis",
            value=st.session_state.evaluation_enabled,
            help=("Get detailed metrics about response quality and " "performance"),
        )
        st.session_state.evaluation_enabled = evaluation_enabled

        if evaluation_enabled:
            st.success("🟢 Advanced Analytics: ON")
            st.caption("📈 Full metrics with quality analysis")
        else:
            st.info("🔴 Quick Mode: ON")
            st.caption("⚡ Faster responses without detailed metrics")

        # Enhanced Performance Metrics
        if st.session_state.evaluation_enabled and st.session_state.performance_data:
            st.markdown("#### 📈 Performance Analytics")

            df = pd.DataFrame(st.session_state.performance_data)

            # Create enhanced charts
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["relevance"],
                    mode="lines+markers",
                    name="Relevance",
                    line=dict(color="#667eea", width=3),
                    marker=dict(size=6),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["faithfulness"],
                    mode="lines+markers",
                    name="Faithfulness",
                    line=dict(color="#764ba2", width=3),
                    marker=dict(size=6),
                )
            )

            fig.update_layout(
                title="Quality Metrics Over Time",
                xaxis_title="Time",
                yaxis_title="Score",
                height=300,
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Performance Summary
            avg_latency = df["latency"].mean()
            avg_relevance = df["relevance"].mean()
            avg_faithfulness = df["faithfulness"].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("⚡ Avg Latency", f"{avg_latency:.2f}s")
            with col2:
                avg_quality = (avg_relevance + avg_faithfulness) / 2
                st.metric("🎯 Avg Quality", f"{avg_quality:.2f}")

    # Main Content
    if not (api_key or provider_key == "huggingface"):
        st.markdown(
            """
            <div class="upload-section">
                <h2>🔐 Welcome to Mini RAG System!</h2>
                <p>Please select a model provider and configure it in the sidebar to begin</p>
                <p>�� Upload documents • 📊 Ask questions • 📊 Get insights</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Enhanced Document Upload Section
    st.markdown("### 📁 Document Management Center")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <div class="upload-section">
                <p>Drag and drop files below or click to browse</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "txt", "csv", "doc", "docx"],
            help="Supported: PDF, TXT, CSV, DOC, DOCX • Max 200MB per file",
            label_visibility="collapsed",
        )

        st.markdown("#### 📝 Or Enter Text Directly")
        text_input = st.text_area(
            "Paste your content here",
            height=120,
            placeholder="Enter your text content here...",
            help="Paste any text content for instant analysis",
        )

    with col2:
        st.markdown("#### ⚡ Quick Actions")

        process_button = st.button(
            "🚀 Process Documents", type="primary", use_container_width=True
        )

        if process_button:
            if uploaded_files or text_input.strip():
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Load documents
                    status_text.text("Loading documents...")
                    progress_bar.progress(25)
                    documents = st.session_state.rag_system.load_documents(
                        uploaded_files, text_input
                    )
                    st.success(f"Loaded {len(documents)} documents")

                    # Chunk documents
                    status_text.text("Processing document chunks...")
                    progress_bar.progress(50)
                    chunks = st.session_state.rag_system.chunk_documents(
                        documents, chunk_size, chunk_overlap
                    )
                    st.success(f"Created {len(chunks)} chunks")

                    # Create vector store
                    status_text.text("Building AI knowledge base...")
                    progress_bar.progress(75)
                    st.session_state.rag_system.create_vectorstore(chunks)
                    st.success("AI knowledge base ready")

                    # Setup QA chain with provider-specific handling
                    status_text.text("Initializing AI assistant...")
                    progress_bar.progress(100)
                    st.session_state.rag_system.setup_qa_chain(
                        selected_model, temperature
                    )

                    st.session_state.documents_loaded = True
                    status_text.empty()
                    progress_bar.empty()

                    st.markdown(
                        create_status_card(
                            "System Ready! Start asking questions below.", "🎉"
                        ),
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    # Clean error message to avoid Unicode issues
                    error_msg = str(e).encode("ascii", "ignore").decode("ascii")
                    st.error(f"Error processing documents: {error_msg}")
            else:
                st.warning("Please upload files or enter text first.")

        if st.session_state.documents_loaded:
            st.divider()
            clear_button = st.button(
                "🗑️ Clear All Data", type="secondary", use_container_width=True
            )

            if clear_button:
                st.session_state.rag_system = RAGSystem()
                # Re-initialize with current provider
                if provider_key == "openai" and api_key:
                    st.session_state.rag_system.initialize_embeddings(
                        api_key, embedding_model
                    )
                elif provider_key == "huggingface":
                    st.session_state.rag_system.initialize_hf_embeddings(
                        embedding_model
                    )
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.session_state.performance_data = []
                st.rerun()

    # Enhanced Chat Interface
    if st.session_state.documents_loaded:
        st.markdown("### 💬 AI Assistant Chat")

        # Chat Container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Display chat history with enhanced styling
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<strong>👤 You:</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message">'
                    f"<strong>🤖 Assistant:</strong><br>"
                    f'{message["content"]}</div>',
                    unsafe_allow_html=True,
                )

                # Enhanced metrics display
                if "metrics" in message and st.session_state.evaluation_enabled:
                    metrics = message["metrics"]

                    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(
                            create_metric_card(
                                "Relevance", f"{metrics['relevance']:.2f}", "🎯"
                            ),
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            create_metric_card(
                                "Faithfulness", f"{metrics['faithfulness']:.2f}", "🔍"
                            ),
                            unsafe_allow_html=True,
                        )
                    with col3:
                        st.markdown(
                            create_metric_card(
                                "Response Time", f"{metrics['latency']:.2f}s", "⚡"
                            ),
                            unsafe_allow_html=True,
                        )
                    with col4:
                        readability_score = metrics["readability"][
                            "flesch_reading_ease"
                        ]
                        st.markdown(
                            create_metric_card(
                                "Readability", f"{readability_score:.1f}", "📖"
                            ),
                            unsafe_allow_html=True,
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Enhanced Query Input
        st.markdown("#### 💭 Ask me anything about your documents:")

        col1, col2 = st.columns([5, 1])
        with col1:
            user_question = st.text_input(
                "Your question",
                value=st.session_state.question_input,
                placeholder="What insights can you provide?",
                label_visibility="collapsed",
                key="question_input_field",
            )
            # Update session state with current input
            st.session_state.question_input = user_question
        with col2:
            ask_button = st.button("🚀 Ask", use_container_width=True)

        # Process question only when button is clicked
        if ask_button and user_question:
            # Clear the input field immediately
            st.session_state.question_input = ""

            # Add user message to chat history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )

            # Create container for streaming response
            response_container = st.empty()
            start_time = time.time()

            try:
                # Create streaming handler
                streaming_handler = StreamingHandler(response_container)

                # Get response
                response = st.session_state.rag_system.get_response(
                    user_question, streaming_handler
                )

                # Calculate latency
                latency = time.time() - start_time

                # Extract answer and source documents
                answer = response["answer"]
                source_docs = response.get("source_documents", [])

                # Prepare message data
                message_data = {
                    "role": "assistant",
                    "content": answer,
                    "sources": [
                        doc.metadata.get("source", "Unknown") for doc in source_docs
                    ],
                    "latency": latency,
                }

                # Calculate evaluation metrics only if enabled
                if st.session_state.evaluation_enabled:
                    relevance = st.session_state.evaluation_metrics.calculate_relevance(
                        user_question, answer, source_docs
                    )
                    faithfulness = (
                        st.session_state.evaluation_metrics.calculate_faithfulness(
                            answer, source_docs
                        )
                    )
                    readability = (
                        st.session_state.evaluation_metrics.calculate_readability(
                            answer
                        )
                    )

                    # Store metrics
                    metrics = {
                        "relevance": relevance,
                        "faithfulness": faithfulness,
                        "latency": latency,
                        "readability": readability,
                    }

                    message_data["metrics"] = metrics

                    # Add to performance data
                    st.session_state.performance_data.append(
                        {
                            "timestamp": datetime.now(),
                            "relevance": relevance,
                            "faithfulness": faithfulness,
                            "latency": latency,
                            "question": user_question,
                        }
                    )

                # Add assistant message to chat history
                st.session_state.chat_history.append(message_data)

                # Enhanced source documents display
                if source_docs:
                    with st.expander("📚 Source Documents", expanded=False):
                        for i, doc in enumerate(source_docs):
                            st.markdown(
                                f"""
                                <div class="source-doc">
                                    <strong>📄 Source {i+1}:</strong> 
                                    {doc.metadata.get('source', 'Unknown')}
                                    <br><br>
                                    <em>{doc.page_content[:200]}...</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Rerun to refresh the display and clear the input
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")

    # Enhanced Sample Questions Section
    if st.session_state.documents_loaded:
        st.markdown(
            """
            <div class="sample-questions">
                <h3>💡 Sample Questions to Get Started</h3>
                <p>Click on any question below to try it out!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        sample_questions = [
            {
                "question": "What are the main topics covered?",
                "icon": "📋",
                "description": "Get an overview of key themes",
            },
            {
                "question": "Can you summarize the key findings?",
                "icon": "📊",
                "description": "Extract important insights",
            },
            {
                "question": "What are the most important points?",
                "icon": "⭐",
                "description": "Highlight critical information",
            },
        ]

        for i, q_data in enumerate(sample_questions):
            with [col1, col2, col3][i]:
                if st.button(
                    f"{q_data['icon']} {q_data['question']}",
                    key=f"sample_{i}",
                    help=q_data["description"],
                    use_container_width=True,
                ):
                    # Add the sample question directly to chat and process it
                    st.session_state.chat_history.append(
                        {"role": "user", "content": q_data["question"]}
                    )

                    # Process the sample question immediately
                    try:
                        start_time = time.time()
                        response = st.session_state.rag_system.get_response(
                            q_data["question"], None
                        )

                        latency = time.time() - start_time
                        answer = response["answer"]
                        source_docs = response.get("source_documents", [])

                        message_data = {
                            "role": "assistant",
                            "content": answer,
                            "sources": [
                                doc.metadata.get("source", "Unknown")
                                for doc in source_docs
                            ],
                            "latency": latency,
                        }

                        if st.session_state.evaluation_enabled:
                            relevance = (
                                st.session_state.evaluation_metrics.calculate_relevance(
                                    q_data["question"], answer, source_docs
                                )
                            )
                            faithfulness = st.session_state.evaluation_metrics.calculate_faithfulness(
                                answer, source_docs
                            )
                            readability = st.session_state.evaluation_metrics.calculate_readability(
                                answer
                            )

                            metrics = {
                                "relevance": relevance,
                                "faithfulness": faithfulness,
                                "latency": latency,
                                "readability": readability,
                            }
                            message_data["metrics"] = metrics

                            st.session_state.performance_data.append(
                                {
                                    "timestamp": datetime.now(),
                                    "relevance": relevance,
                                    "faithfulness": faithfulness,
                                    "latency": latency,
                                    "question": q_data["question"],
                                }
                            )

                        st.session_state.chat_history.append(message_data)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error with sample question: {str(e)}")


if __name__ == "__main__":
    main()
