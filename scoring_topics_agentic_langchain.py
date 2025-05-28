import os
import pandas as pd
import json
import re
import time
import fitz
import logging
import pathlib
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.schema import Document
import requests
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.output_parsers import PydanticOutputParser, ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.tools import Tool, BaseTool, StructuredTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class CGSConfig:
    """Configuration for Corporate Governance Scoring system"""
    
    def __init__(self, company_sym: str, base_path: str = None):
        """Initialize configuration"""
        self.company_sym = company_sym
        if base_path:
            self.base_path = base_path
        else:
            self.base_path = f'/Users/monilshah/Documents/GitHub/AgentEval/{company_sym}/'
        
        # Define paths
        self.data_path = os.path.join(self.base_path, '98_data/')
        self.static_path = os.path.join(self.base_path, '97_static/')
        self.results_path = os.path.join(self.base_path, '96_results/')
        self.parent_path = os.path.dirname(os.path.dirname(self.base_path))
        
        # Ensure directories exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.static_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        # Model configuration
        # self.model_to_use = 'llama3'
        # Model configuration
        self.model_provider = "ollama"  # Options: "ollama", "gemini"
        self.model_to_use = 'llama3'    # For Ollama
        self.gemini_model = "gemini-1.5-flash"  # For Gemini
        # Alternative: self.model_to_use = 'deepseek-R1'
        
        # API keys - in a production environment, these should be loaded securely
        self.configure_api_keys()
        
        # Retrieval configuration
        self.retrieval_method = "hybrid"  # Options: "hybrid", "bm25", "vector", "direct"
        self.bm25_weight = 0.5
        self.vector_weight = 0.5

    def configure_api_keys(self):
        """Configure API keys from environment or .env file"""
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()  # Take environment variables from .env
        
        # Now get API keys from environment
        google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        chatpdf_api_key = os.environ.get("CHATPDF_API_KEY", "")
        
        # Set the API keys
        if not google_api_key:
            logger.warning("GOOGLE_API_KEY not found in environment or .env file")
        else:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            
        self.CHATPDF_API_KEY = chatpdf_api_key
        if not chatpdf_api_key:
            logger.warning("CHATPDF_API_KEY not found in environment or .env file")
        
        # Initialize API clients if needed
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            self.genai_client = genai
            logger.info("Google Generative AI client initialized successfully")
        except ImportError:
            logger.warning("Google Generative AI package not installed")
            self.genai_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Google Generative AI client: {e}")
            self.genai_client = None
                     
class DocumentProcessor:
    """Handles document processing operations: downloading, splitting, and loading PDFs"""
    
    def __init__(self, config: CGSConfig):
        self.config = config
        self.source_to_path_map = {}
        self.vector_stores = {}
    
    def download_pdfs_from_csv(self):
        """Download PDFs from URLs listed in the source_url.csv file"""
        csv_path = os.path.join(self.config.static_path, 'source_url.csv')
        if not os.path.exists(csv_path):
            logger.info('No source url file found')
            return {}
        
        df = pd.read_csv(csv_path)
        ids = {}
        for _, row in df.iterrows():
            source_id_name = row['source']
            url_link = row['url']
            save_path = os.path.join(self.config.data_path, f"{source_id_name}.pdf")
            
            try:
                # Download PDF
                filepath = pathlib.Path(save_path)
                filepath.write_bytes(httpx.get(url_link).content)
                logger.info(f"Downloaded PDF from {url_link} and saved as {filepath}")
                ids[source_id_name] = save_path
            except Exception as e:
                logger.error(f"Failed to download {url_link}: {e}")
        
        return ids
    
    def split_large_pdfs(self, size_limit_mb=15, overlap_pages=10):
        """Split PDFs that exceed the size limit"""
        all_files = [f for f in os.listdir(self.config.data_path) if f.endswith('.pdf')]
        all_files = [os.path.join(self.config.data_path, f) for f in all_files]
        
        for file_path in all_files:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > size_limit_mb:
                logger.info(f"Splitting large PDF: {file_path} ({file_size_mb:.2f} MB)")
                self._split_pdf(file_path, overlap_pages)
            else:
                logger.debug(f"PDF is under size limit: {file_path} ({file_size_mb:.2f} MB)")
    
    def _split_pdf(self, input_pdf, overlap_pages=10):
        """Split a PDF into two parts with overlap"""
        try:
            doc = fitz.open(input_pdf)
            total_pages = len(doc)
            
            if total_pages < 2:
                logger.info(f"Skipping {input_pdf}: Not enough pages to split")
                return
            
            # Calculate midpoint
            mid = total_pages // 2
            
            # Generate new file names
            base_name, ext = os.path.splitext(input_pdf)
            output_pdf1 = f"{base_name}_1{ext}"
            output_pdf2 = f"{base_name}_2{ext}"
            
            # Create first half
            doc1 = fitz.open()
            doc1.insert_pdf(doc, from_page=0, to_page=mid - 1)
            doc1.save(output_pdf1)
            doc1.close()
            
            # Create second half with overlap
            doc2 = fitz.open()
            doc2.insert_pdf(doc, from_page=max(0, mid-overlap_pages), to_page=total_pages - 1)
            doc2.save(output_pdf2)
            doc2.close()
            
            if os.path.exists(output_pdf1) and os.path.exists(output_pdf2):
                os.remove(input_pdf)
                logger.info(f"PDF split successfully and original deleted:\n  {output_pdf1} ({mid} pages)\n  {output_pdf2} ({total_pages - mid} pages)")
            else:
                logger.error(f"Error: Splitting {input_pdf} failed, original file not deleted")
        except Exception as e:
            logger.error(f"Error splitting PDF {input_pdf}: {e}")
    
    def create_source_map(self):
        """Create mapping between source IDs and file paths"""
        all_files = [f for f in os.listdir(self.config.data_path) if f.endswith('.pdf')]
        all_files = [os.path.join(self.config.data_path, f) for f in all_files]
        
        if not all_files:
            logger.info('No PDF files found in data directory')
            return {}
        
        ids = {}
        for file_path in all_files:
            source_id_name = os.path.basename(file_path).split('.')[0]
            ids[source_id_name] = file_path
        
        # Save mapping to CSV
        save_path = os.path.join(self.config.static_path, 'source_path_map.csv')
        df_ids = pd.DataFrame(ids.items(), columns=['source', 'path'])
        df_ids.to_csv(save_path, index=False)
        logger.info(f'Source map saved to {save_path}')
        
        self.source_to_path_map = ids
        return ids
    
    def create_vector_store(self, document_path, chunk_size=None, chunk_overlap=None):
        """Create a lightweight vector store using page-level chunks"""
        try:
            # Check if we already have a vector store for this document
            if document_path in self.vector_stores:
                return self.vector_stores[document_path]
            
            # Use page-level chunks instead of arbitrary token-based chunking
            page_chunks = self.create_page_level_chunks(document_path)
            
            # Use a lighter embedding model if possible
            try:
                # First try a local, lightweight model
                #embeddings = OllamaEmbeddings(model="llama3")  # Smaller embedding dimension
                embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': True}
                        )
            except Exception as e:
                logger.warning(f"Failed to initialize Huggingface embeddings: {e}.")
                #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Lighter model
            
            # Use in-memory store for small documents, persistent for larger ones
            doc_id = os.path.basename(document_path).replace('.pdf', '')
            persist_directory = os.path.join(self.config.static_path, f"vector_store_{doc_id}")
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=page_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            
            # Store reference to vector store
            self.vector_stores[document_path] = vector_store
            logger.info(f"Created lightweight vector store for {document_path} with {len(page_chunks)} page chunks")
            
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store for {document_path}: {e}")
            return None

    def create_hybrid_retriever(self, document_path):
        """Create a hybrid retriever combining BM25 and semantic search"""
        # Check if we already have a vector store for this document
        vector_store = self.create_vector_store(document_path)
        if not vector_store:
            logger.error(f"Failed to create vector store for {document_path}")
            return None
        
        # Load document for BM25
        try:
            loader = PyPDFLoader(document_path)
            documents = loader.load()
            
            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(documents)
            
            # Get vector retriever from existing vector store
            vector_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create ensemble retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]  # Adjust weights based on performance
            )
            
            logger.info(f"Created hybrid retriever for {document_path}")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Error creating hybrid retriever: {e}")
            return None
       
    def create_page_level_chunks(self, document_path):
        """Create page-level chunks with rich metadata"""
        try:
            doc = fitz.open(document_path)
            file_name = os.path.basename(document_path)
            
            chunks = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Create a document with rich metadata
                chunk = Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "page": page_num + 1,
                        "file_path": document_path,
                        "total_pages": len(doc),
                        "chunk_type": "page"
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} page-level chunks for {document_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error creating page-level chunks: {e}")
            return []
    
    def create_retriever(self, document_path):
        """Create a retriever based on configuration settings"""
        try:
            # Check if direct method is specified
            retrieval_method = self.config.retrieval_method.lower()
            if retrieval_method == "direct":
                logger.info(f"Direct method specified. No retriever needed for {document_path}")
                return None  # No retriever needed for direct method
                
            # Load document
            loader = PyPDFLoader(document_path)
            documents = loader.load()
            
            if not documents:
                logger.error(f"No content found in {document_path}")
                return None
                
            # Create page-level chunks
            page_chunks = self.create_page_level_chunks(document_path)
            
            # Determine which retriever to create based on config
            if retrieval_method == "bm25" or retrieval_method == "hybrid":
                # Create BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 10  # Return more results for re-ranking
                
                if retrieval_method == "bm25":
                    logger.info(f"Created BM25-only retriever for {document_path}")
                    return bm25_retriever
            
            if retrieval_method == "vector" or retrieval_method == "hybrid":
                # Create or get vector store
                vector_store = self.create_vector_store(document_path)
                if not vector_store:
                    logger.warning(f"Failed to create vector store for {document_path}")
                    
                    # Fallback to BM25 if vector store creation failed
                    if retrieval_method == "hybrid" and 'bm25_retriever' in locals():
                        logger.info(f"Falling back to BM25-only retriever for {document_path}")
                        return bm25_retriever
                    return None
                    
                # Create vector retriever
                vector_retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}  # Return more results for re-ranking
                )
                
                if retrieval_method == "vector":
                    logger.info(f"Created vector-only retriever for {document_path}")
                    return vector_retriever
            
            # Create hybrid retriever if we reached here
            if retrieval_method == "hybrid" and 'bm25_retriever' in locals() and 'vector_retriever' in locals():
                hybrid_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[self.config.bm25_weight, self.config.vector_weight]
                )
                logger.info(f"Created hybrid retriever for {document_path} with weights [BM25: {self.config.bm25_weight}, Vector: {self.config.vector_weight}]")
                return hybrid_retriever
                
            # If we reach here, something went wrong
            logger.error(f"Failed to create retriever with method {retrieval_method}")
            return None
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            return None
    
class QueryEngine:
    """Handles document querying using different models and approaches"""
    
    def __init__(self, config: CGSConfig, document_processor: DocumentProcessor):
        self.config = config
        self.document_processor = document_processor
        self.chatpdf_ids = {}  # Source path to ChatPDF ID mapping
        
        # Initialize LLM clients
        self.setup_llm_clients()
    
    def query_document(self, document_path: str, prompt: str, use_rag: bool = True, return_chunks: bool = False):
        """Query a document with a given prompt, optionally returning chunks for guardrail validation"""
        logger.info(f"Querying document {document_path} with prompt: {prompt}")
        
        try:
            # Check if we're using "direct" retrieval method
            if self.config.retrieval_method.lower() == "direct":
                # Skip RAG and go directly to Gemini query
                logger.info(f"Using direct method as configured, bypassing RAG")
                result = self._query_with_gemini(document_path, prompt)
                if result and len(result.strip()) > 0:
                    # Return result and None for chunks since direct method doesn't use RAG
                    return (result, None) if return_chunks else result
                    
                # If that fails, try ChatPDF as final fallback
                if self.config.CHATPDF_API_KEY:
                    result = self._query_with_chatpdf(document_path, prompt)
                    return (result, None) if return_chunks else result
                    
                fallback_result = "Unable to process document query with direct method."
                return (fallback_result, None) if return_chunks else fallback_result
                
            # For non-direct methods, proceed with RAG if enabled
            if use_rag:
                rag_result = self._query_with_rag(document_path, prompt, return_chunks=return_chunks)
                
                if return_chunks:
                    result, top_chunks = rag_result if isinstance(rag_result, tuple) else (rag_result, None)
                else:
                    result = rag_result
                    top_chunks = None
                
                if result and len(result.strip()) > 0:
                    return (result, top_chunks) if return_chunks else result
            
            # Fallback to direct Gemini query
            result = self._query_with_gemini(document_path, prompt)
            if result and len(result.strip()) > 0:
                return (result, None) if return_chunks else result
            
            # Final fallback to ChatPDF if configured
            if self.config.CHATPDF_API_KEY:
                result = self._query_with_chatpdf(document_path, prompt)
                return (result, None) if return_chunks else result
                
            fallback_result = "Unable to process document query."
            return (fallback_result, None) if return_chunks else fallback_result
        except Exception as e:
            logger.error(f"Error querying document: {e}")
            error_result = f"Error querying document: {str(e)}"
            return (error_result, None) if return_chunks else error_result
    
    def _query_with_rag(self, document_path, prompt, return_chunks=False):
        """Query using optimized hybrid RAG with PDF slices, optionally returning chunks"""
        try:
            # Check if we're using direct method
            if self.config.retrieval_method.lower() == "direct":
                logger.info(f"Direct method specified. Bypassing RAG for {document_path}")
                return (None, None) if return_chunks else None
                
            # Store the current document path for fallback
            self.current_document_path = document_path
            
            # Create page-level chunks if not already cached
            chunks_cache = getattr(self, 'chunks_cache', {})
            if document_path not in chunks_cache:
                chunks_cache[document_path] = self.document_processor.create_page_level_chunks(document_path)
            self.chunks_cache = chunks_cache
            
            # Create retriever if not already cached
            retriever_cache = getattr(self, 'retriever_cache', {})
            if document_path not in retriever_cache:
                retriever_cache[document_path] = self.document_processor.create_retriever(document_path)
            self.retriever_cache = retriever_cache
            
            retriever = retriever_cache[document_path]
            if not retriever:
                logger.warning(f"No retriever available for {document_path}")
                return (None, None) if return_chunks else None
            
            # Get relevant documents using the modern API
            try:
                # First try the new invoke method
                top_chunks = retriever.invoke(prompt)
                print(f"Top chunks retrieved: {len(top_chunks)} for sending to gemini")
            except (AttributeError, TypeError):
                # Fall back to the legacy method if needed
                logger.warning("Falling back to legacy get_relevant_documents method")
                try:
                    top_chunks = retriever.get_relevant_documents(prompt)
                except Exception as e:
                    logger.error(f"Error retrieving documents: {e}")
                    # Final fallback: use the first few chunks as a last resort
                    if document_path in chunks_cache and chunks_cache[document_path]:
                        logger.warning("Using first few chunks as fallback")
                        top_chunks = chunks_cache[document_path][:5]  # Use first 5 chunks
                    else:
                        logger.error("No chunks available for fallback")
                        return (None, None) if return_chunks else None
            
            if not top_chunks:
                logger.warning(f"No relevant chunks found for query in {document_path}")
                
                # Fallback to using first few chunks if we have them cached
                if document_path in chunks_cache and chunks_cache[document_path]:
                    logger.info("Using first few chunks as fallback since no relevant chunks found")
                    top_chunks = chunks_cache[document_path][:5]  # Use first 5 chunks as fallback
                else:
                    return (None, None) if return_chunks else None
            
            # Limit to at most 10 chunks to keep processing time reasonable
            if len(top_chunks) > 10:
                top_chunks = top_chunks[:10]
            
            # Query with PDF slices
            result = self.query_with_pdf_slices(prompt, top_chunks)
            print(f"RAG query result: {result}")
            
            logger.info(f"RAG query completed for {document_path} with {len(top_chunks)} chunks")
            
            # Return result and chunks if requested
            if return_chunks:
                return result, top_chunks
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (None, None) if return_chunks else None
     
    def setup_llm_clients(self):
        """Set up LLM clients for querying"""
        try:
            # Check which provider is configured
            if self.config.model_provider == "ollama":
                try:
                    # Initialize Ollama client
                    self.ollama = OllamaLLM(model=self.config.model_to_use)
                    logger.info(f"Initialized Ollama with model {self.config.model_to_use}")
                except Exception as e:
                    logger.error(f"Error initializing Ollama: {e}")
                    logger.warning("Falling back to Gemini for all operations")
                    self.config.model_provider = "gemini"
            
            # Always initialize Gemini client (used for PDF processing regardless of provider)
            self.gemini = ChatGoogleGenerativeAI(model=self.config.gemini_model)
            logger.info(f"Initialized Gemini client with model {self.config.gemini_model}")
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {e}")
     

    def query_gemini_with_pdf(self, pdf_path, prompt, max_size_mb=20, is_temp_file=False):
        """
        Query Gemini with a PDF document and a prompt
        
        Parameters:
        - pdf_path: Path to the PDF file
        - prompt: The query prompt to send along with the PDF
        - max_size_mb: Maximum PDF size in MB (default: 20MB)
        - is_temp_file: Whether this is a temporary file that should be deleted after use
        
        Returns:
        - Gemini's response text, or an error message if the query fails
        """
        try:
            if not self.config.genai_client:
                return "Google API client not initialized."
            
            # Check file size - Gemini has limitations
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                logger.warning(f"Document too large for Gemini: {file_size_mb:.2f} MB > {max_size_mb} MB")
                
                # For large files, we could implement alternative strategies:
                # 1. Try with first N pages
                # 2. Compress the PDF
                # 3. Extract and send text only
                return f"Document size ({file_size_mb:.2f} MB) exceeds Gemini's limit ({max_size_mb} MB)"
            
            # Prepare the query
            genai = self.config.genai_client
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Read file as bytes
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
                
            # Log basic information
            logger.info(f"Sending PDF to Gemini - Size: {file_size_mb:.2f}MB, File: {os.path.basename(pdf_path)}")
            
            # Create a multimodal prompt
            import base64
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                    "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    prompt
                ]
            )
            
            result = response.text
            
            # Clean up temporary file if requested
            if is_temp_file:
                try:
                    os.remove(pdf_path)
                    logger.debug(f"Removed temporary PDF: {pdf_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary PDF: {cleanup_error}")
            
            return result
        except Exception as e:
            logger.error(f"Error querying Gemini with PDF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error processing PDF with Gemini: {str(e)}"
    
    def _query_with_gemini(self, document_path, prompt):
        """Query directly with Gemini and the full document"""
        try:
            if not self.config.genai_client:
                logger.warning("Gemini client not initialized")
                return None
                    
            # Check file size - Gemini has limitations
            file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
            if file_size_mb > 20:
                logger.warning(f"Document too large for direct Gemini query: {file_size_mb:.2f} MB")
                return None
            
            # Prepare the query using the module reference
            genai = self.config.genai_client
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Read file as bytes
            with open(document_path, 'rb') as f:
                file_bytes = f.read()
                
            # Create a multipart prompt using base64 encoding for the PDF bytes
            import base64
            
            # Log that we're sending the actual PDF, not just text
            logger.info(f"Sending full PDF document to Gemini: {os.path.basename(document_path)}")
            
            # Send the document directly to Gemini - multimodal approach
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                    "data": base64.b64encode(file_bytes).decode('utf-8')},
                    prompt
                ]
            )
            
            return response.text
        except Exception as e:
            if "Request payload size exceeds the limit" in str(e):
                logger.warning(f"Document too large for Gemini: {e}")
            else:
                logger.error(f"Error querying Gemini: {e}")
                import traceback
                logger.error(traceback.format_exc())
            return None
    
    def _upload_to_chatpdf(self, document_path):
        """Upload a document to ChatPDF and get source ID"""
        try:
            # Check if ChatPDF API key is configured
            if not self.config.CHATPDF_API_KEY:
                logger.warning("ChatPDF API key not configured")
                return None
                
            # Check if already uploaded
            if document_path in self.chatpdf_ids:
                return self.chatpdf_ids[document_path]
            
            # Load existing mappings if available
            chatpdf_map_path = os.path.join(self.config.static_path, 'chatpdf_source_id_map.csv')
            if os.path.exists(chatpdf_map_path):
                chatpdf_df = pd.read_csv(chatpdf_map_path)
                for _, row in chatpdf_df.iterrows():
                    self.chatpdf_ids[row['source']] = row['source_id']
                
                if document_path in self.chatpdf_ids:
                    return self.chatpdf_ids[document_path]
            
            # Upload to ChatPDF
            files = [
                ('file', ('file', open(document_path, 'rb'), 'application/octet-stream'))
            ]
            headers = {
                'x-api-key': self.config.CHATPDF_API_KEY
            }
            
            response = requests.post(
                'https://api.chatpdf.com/v1/sources/add-file', 
                headers=headers, 
                files=files
            )
            
            if response.status_code == 200:
                source_id = response.json()['sourceId']
                self.chatpdf_ids[document_path] = source_id
                
                # Save updated mappings
                ids_df = pd.DataFrame([[document_path, source_id]], columns=['source', 'source_id'])
                if os.path.exists(chatpdf_map_path):
                    existing_df = pd.read_csv(chatpdf_map_path)
                    ids_df = pd.concat([existing_df, ids_df], ignore_index=True)
                
                ids_df.to_csv(chatpdf_map_path, index=False)
                logger.info(f"Uploaded {document_path} to ChatPDF with ID {source_id}")
                return source_id
            else:
                logger.error(f"ChatPDF upload failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error uploading to ChatPDF: {e}")
            return None
    
    def query_with_pdf_slices(self, prompt, top_chunks):
        """Query Gemini with slices of the original PDF, including context pages"""
        try:
            # Store the current document path for fallback
            current_doc = getattr(self, 'current_document_path', None)
            
            # Try to extract file paths and page numbers in a format-independent way
            pdf_slices = []
            
            for chunk in top_chunks:
                # Start with an empty info dict
                slice_info = {"page": 1}  # Default to page 1
                
                # Check if it's a Document object with metadata
                if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                    metadata = chunk.metadata
                    # Try different possible keys for file path
                    for key in ['file_path', 'source', 'path', 'filename']:
                        if key in metadata and metadata[key]:
                            file_path = metadata[key]
                            # If it's just a filename, prepend the data path
                            if os.path.dirname(file_path) == '':
                                file_path = os.path.join(self.config.data_path, file_path)
                            slice_info["file_path"] = file_path
                            break
                    
                    # Try different possible keys for page number
                    for key in ['page', 'page_number', 'page_num']:
                        if key in metadata and metadata[key]:
                            try:
                                slice_info["page"] = int(metadata[key])
                            except (ValueError, TypeError):
                                pass
                            break
                
                # If we couldn't find a file path, try a different approach
                if "file_path" not in slice_info and current_doc:
                    # Use the document path we're currently querying
                    slice_info["file_path"] = current_doc
                
                # Add to our slices if we have a file path
                if "file_path" in slice_info and os.path.exists(slice_info["file_path"]):
                    pdf_slices.append(slice_info)
                else:
                    logger.warning(f"Could not determine valid file path for chunk")
            
            # If we couldn't extract any valid slices, use a fallback approach
            if not pdf_slices:
                logger.warning("No valid file paths found in chunks, using current document")
                if current_doc and os.path.exists(current_doc):
                    # Create default slices for the first few pages of the document
                    doc = fitz.open(current_doc)
                    num_pages = min(5, len(doc))  # Use at most 5 pages
                    pdf_slices = [{"file_path": current_doc, "page": i+1} for i in range(num_pages)]
                else:
                    return "Could not locate valid document sources for query."
            
            # Group slices by file path
            files_to_pages = {}
            for s in pdf_slices:
                file_path = s['file_path']
                page = int(s['page'])
                
                if file_path not in files_to_pages:
                    files_to_pages[file_path] = set()
                
                # Add the page and its context (one before and one after)
                files_to_pages[file_path].add(page - 1)  # One page before
                files_to_pages[file_path].add(page)      # The actual page
                files_to_pages[file_path].add(page + 1)  # One page after
            
            # Create a temporary PDF with the pages and their context
            output_pdf = fitz.open()
            
            # Keep track of pages added for citation
            added_pages = {}
            
            for file_path, pages in files_to_pages.items():
                # Open the source document
                try:
                    doc = fitz.open(file_path)
                    total_pages = len(doc)
                    file_name = os.path.basename(file_path)
                    
                    if file_name not in added_pages:
                        added_pages[file_name] = []
                    
                    # Sort pages and filter out invalid page numbers
                    valid_pages = sorted([p for p in pages if 0 <= p < total_pages])
                    
                    for page_num in valid_pages:
                        # Add the page to the new PDF
                        output_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                        
                        # Record original page number for citation (pages are 1-indexed in citations)
                        added_pages[file_name].append(page_num + 1)
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
            
            if output_pdf.page_count == 0:
                logger.error("Failed to create PDF with relevant pages")
                return "Could not extract relevant document pages for analysis."
            
            # Save temp PDF
            import time
            import tempfile
            temp_pdf_path = os.path.join(tempfile.gettempdir(), f"temp_slice_{int(time.time())}.pdf")
            output_pdf.save(temp_pdf_path)
            
            # Query Gemini with the PDF using direct multimodal approach
            genai = self.config.genai_client
            if not genai:
                return "Google API client not initialized."
                
            import base64
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Read file as bytes for multimodal query
            with open(temp_pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Log what we're doing
            logger.info(f"Sending sliced PDF to Gemini with {output_pdf.page_count} pages")
            
            # Send the PDF directly to Gemini as multimodal content
            try:
                response = model.generate_content(
                    contents=[
                        {"mime_type": "application/pdf", 
                        "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                        prompt
                    ]
                )
                
                result = response.text
            except Exception as gemini_error:
                logger.error(f"Gemini API error: {gemini_error}")
                return f"Error querying Gemini: {str(gemini_error)}"
            
            # Create page range citations
            citations = []
            for file_name, pages in added_pages.items():
                if not pages:
                    continue
                    
                # Sort pages
                pages = sorted(list(set(pages)))  # Ensure unique, sorted list
                
                if not pages:  # Skip if empty after filtering
                    continue
                    
                # Group consecutive pages into ranges
                ranges = []
                if len(pages) > 0:
                    range_start = pages[0]
                    prev_page = pages[0]
                    
                    for page in pages[1:]:
                        if page > prev_page + 1:
                            # End of consecutive range
                            if range_start == prev_page:
                                ranges.append(f"{range_start}")
                            else:
                                ranges.append(f"{range_start}-{prev_page}")
                            range_start = page
                        prev_page = page
                    
                    # Add the last range
                    if range_start == prev_page:
                        ranges.append(f"{range_start}")
                    else:
                        ranges.append(f"{range_start}-{prev_page}")
                
                # Add citation for this file
                citations.append(f"pp. {', '.join(ranges)} ({file_name})")
            
            # Add citations to result
            if citations:
                result += f"\n\nSources: {'; '.join(citations)}"
            
            # Clean up
            try:
                os.remove(temp_pdf_path)
                logger.debug(f"Removed temporary PDF: {temp_pdf_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary PDF: {cleanup_error}")
            
            return result
        except Exception as e:
            logger.error(f"Error in PDF slice query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error querying with PDF slices: {str(e)}"
    
    def _query_with_chatpdf(self, document_path, prompt):
        """Query using ChatPDF API"""
        try:
            source_id = self._upload_to_chatpdf(document_path)
            if not source_id:
                return None
                
            headers = {
                'x-api-key': self.config.CHATPDF_API_KEY,
                "Content-Type": "application/json",
            }
            
            data = {
                'sourceId': source_id,
                'messages': [
                    {
                        'role': "user",
                        'content': prompt,
                    }
                ]
            }
            
            response = requests.post(
                'https://api.chatpdf.com/v1/chats/message', 
                headers=headers, 
                json=data
            )
            
            if response.status_code == 200:
                result = response.json().get('content', '')
                return result
            else:
                logger.error(f"ChatPDF query failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error in ChatPDF query: {e}")
            return None
        
    def query_with_gemini_multimodal(self, temp_pdf_path, prompt):
        """Query Gemini with a PDF using the correct API format"""
        try:
            genai = self.config.genai_client
            if not genai:
                return "Google API client not initialized."
                    
            import base64
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Read file as bytes
            with open(temp_pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Different approach - using direct content parts
            # This avoids using genai.types.Part which might not exist
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                    "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    prompt
                ]
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error querying Gemini: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback to simple text response if PDF processing fails
            return f"Error processing document with Gemini. The error was: {str(e)}"
                        
class GuardrailAgent:
    """Handles verification of LLM outputs with enhanced context validation"""
    
    def __init__(self, config: CGSConfig):
        self.config = config
        
        # Initialize LLMs based on configuration
        try:
            # Initialize Gemini
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.gemini = ChatGoogleGenerativeAI(model=config.gemini_model)
            logger.info(f"Initialized Gemini client with model {config.gemini_model}")
            
            # Initialize Ollama if that's the selected provider
            if config.model_provider == "ollama":
                try:
                    from langchain_ollama import OllamaLLM
                    self.ollama = OllamaLLM(model=config.model_to_use)
                    logger.info(f"Initialized Ollama with model {config.model_to_use}")
                except Exception as e:
                    logger.error(f"Error initializing Ollama: {e}")
                    logger.warning("Falling back to Gemini for all operations")
                    self.config.model_provider = "gemini"
        except Exception as e:
            logger.error(f"Error initializing LLMs: {e}")
    
    def get_guardrail_llm(self):
        """Get the appropriate LLM for guardrail operations"""
        if self.config.model_provider == "ollama" and hasattr(self, 'ollama'):
            return self.ollama
        else:
            return self.gemini
    
    def verify_answer_quality(self, query: str, answer: str, top_chunks=None) -> Dict[str, str]:
        """Verify if the answer is good quality and has proper citations with enhanced context validation"""
        
        if not hasattr(self, 'gemini'):
            logger.warning("No LLM available for verification, using simplified checks")
            # Simple fallback checks
            got_answer = "no" if "could not process document" in answer.lower() or "error" in answer.lower() else "yes"
            source_mentioned = "yes" if "page" in answer.lower() or "source" in answer.lower() else "no"
            return {"got_answer": got_answer, "source_mentioned": source_mentioned}
        
        # Determine if we should use enhanced verification with chunks
        use_enhanced_verification = (
            top_chunks is not None and 
            len(top_chunks) > 0 and 
            self.config.retrieval_method.lower() != "direct"
        )
        
        if use_enhanced_verification:
            return self._verify_with_chunk_context(query, answer, top_chunks)
        else:
            return self._verify_basic(query, answer)
    
    def _verify_basic(self, query: str, answer: str) -> Dict[str, str]:
        """Basic verification without chunk context (original implementation)"""
        
        # Define the output schema
        class GuardrailOutput(BaseModel):
            got_answer: str = Field(description="Whether the LLM provided a substantive answer to the query (yes/no)")
            source_mentioned: str = Field(description="Whether the answer mentions page numbers or specific sections (yes/no)")
            
        # Create custom format instructions
        custom_format_instructions = """
        You must respond with a valid JSON object using exactly this format:
        {
            "got_answer": "<yes_or_no>",
            "source_mentioned": "<yes_or_no>"
        }
        
        Both values must be either "yes" or "no" (lowercase).
        Do not include any other text, explanation, or formatting outside of this JSON object.
        """
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_template(
            """
            You are a smart LLM output assessor. 
            We asked following query to one LLM: {query}
            
            The LLM came up with the following answer: {answer}
            
            Evaluate the quality of this answer based on two criteria:
            1. Did the LLM actually provide a substantive answer to the query, or did it fail to address the question?
            2. Does the answer include references to specific page numbers, sections, or documents to support its claims?
            
            To be considered reliable, the answer should cite specific sources, such as "From the annual report page XX, we found that..."
            
            {format_instructions}
            
            Remember: Your response must be ONLY a valid JSON object with the keys "got_answer" and "source_mentioned".
            """
        )
        
        try:
            # Get the appropriate LLM
            llm = self.get_guardrail_llm()
            
            # Try the chain approach
            raw_result = llm.invoke(prompt.format(
                query=query,
                answer=answer,
                format_instructions=custom_format_instructions
            ))
            
            logger.info("Received basic guardrail assessment, attempting to parse")
            return self._parse_guardrail_result(raw_result, answer)
                
        except Exception as e:
            logger.error(f"Error in basic guardrail verification: {e}")
                
        # Simple fallback if all else fails
        got_answer = "no" if "could not process document" in answer.lower() or "error" in answer.lower() else "yes"
        source_mentioned = "yes" if "page" in answer.lower() or "source" in answer.lower() else "no"
        
        logger.info(f"Using simple heuristic fallback: got_answer={got_answer}, source_mentioned={source_mentioned}")
        return {"got_answer": got_answer, "source_mentioned": source_mentioned}
    
    def _verify_with_chunk_context(self, query: str, answer: str, top_chunks) -> Dict[str, str]:
        """Enhanced verification using retrieved chunk context"""
        
        # Prepare chunk context for the prompt
        chunk_context = self._prepare_chunk_context(top_chunks)
        
        # Define the enhanced output schema
        class EnhancedGuardrailOutput(BaseModel):
            got_answer: str = Field(description="Whether the LLM provided a substantive answer to the query (yes/no)")
            source_mentioned: str = Field(description="Whether the answer mentions page numbers or specific sections (yes/no)")
            answer_grounded: str = Field(description="Whether the answer is well-grounded in the provided source chunks (yes/no)")
            relevance_score: str = Field(description="How relevant is the answer to the query (high/medium/low)")
            
        # Create custom format instructions
        custom_format_instructions = """
        You must respond with a valid JSON object using exactly this format:
        {
            "got_answer": "<yes_or_no>",
            "source_mentioned": "<yes_or_no>",
            "answer_grounded": "<yes_or_no>",
            "relevance_score": "<high_medium_or_low>"
        }
        
        - got_answer and source_mentioned and answer_grounded must be either "yes" or "no" (lowercase)
        - relevance_score must be either "high", "medium", or "low" (lowercase)
        Do not include any other text, explanation, or formatting outside of this JSON object.
        """
        
        # Create the enhanced prompt
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert document analysis quality assessor. Your job is to evaluate whether an LLM's answer is appropriate given the source material.
            
            ORIGINAL QUERY: {query}
            
            LLM'S ANSWER: {answer}
            
            RETRIEVED SOURCE CHUNKS:
            {chunk_context}
            
            Evaluate the LLM's answer based on these criteria:
            
            1. **got_answer**: Did the LLM provide a substantive, meaningful answer to the query (not just "I don't know" or error messages)?
            
            2. **source_mentioned**: Does the answer include specific references to page numbers, sections, or document sources?
            
            3. **answer_grounded**: Is the answer well-supported by the information in the retrieved chunks? Look for:
               - Claims made in the answer that can be verified in the source chunks
               - Whether the answer goes beyond what's available in the chunks (potential hallucination)
               - Whether key facts mentioned align with the source material
               - Unless you think the answer is completely off-tack, you can consider it grounded.
            
            4. **relevance_score**: How well does the answer address the specific query?
               - "high": Directly and comprehensively answers the question
               - "medium": Partially answers or provides related information
               - "low": Doesn't really address the query or provides irrelevant information
            
            Be strict in your evaluation. If the answer makes claims not supported by the chunks, mark answer_grounded as "no".
            
            {format_instructions}
            """
        )
        
        try:
            # Get the appropriate LLM
            llm = self.get_guardrail_llm()
            
            # Run the enhanced verification
            raw_result = llm.invoke(prompt.format(
                query=query,
                answer=answer,
                chunk_context=chunk_context,
                format_instructions=custom_format_instructions
            ))
            
            logger.info("Received enhanced guardrail assessment with chunk context")
            
            # Parse the enhanced result
            result = self._parse_enhanced_guardrail_result(raw_result, answer)
            
            # Log the enhanced assessment
            logger.info(f"Enhanced guardrail result: {result}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error in enhanced guardrail verification: {e}")
            # Fall back to basic verification
            logger.info("Falling back to basic verification due to enhanced verification error")
            return self._verify_basic(query, answer)
    
    def _prepare_chunk_context(self, top_chunks, max_chunks=5, max_chars_per_chunk=300):
        """Prepare a concise context summary from top chunks"""
        
        if not top_chunks:
            return "No source chunks available."
        
        context_parts = []
        
        # Limit the number of chunks to avoid token limits
        chunks_to_use = top_chunks[:max_chunks]
        
        for i, chunk in enumerate(chunks_to_use):
            chunk_info = f"**Chunk {i+1}:**\n"
            
            # Add metadata if available
            if hasattr(chunk, 'metadata') and chunk.metadata:
                metadata = chunk.metadata
                if 'page' in metadata:
                    chunk_info += f"- Page: {metadata['page']}\n"
                if 'source' in metadata:
                    chunk_info += f"- Source: {metadata['source']}\n"
                if 'file_path' in metadata:
                    chunk_info += f"- File: {os.path.basename(metadata['file_path'])}\n"
            
            # Add content (truncated if necessary)
            content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk] + "..."
            
            chunk_info += f"- Content: {content}\n\n"
            context_parts.append(chunk_info)
        
        return "".join(context_parts)
    
    def _parse_enhanced_guardrail_result(self, raw_result, answer):
        """Parse enhanced guardrail result with fallback to basic fields"""
        
        try:
            # Try multiple parsing approaches (similar to existing logic)
            if hasattr(raw_result, 'content') and isinstance(raw_result.content, str):
                message_content = raw_result.content
                logger.info(f"Parsing enhanced guardrail from AIMessage content")
                
                # Try to extract JSON from the content
                import re
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, message_content, re.DOTALL)
                
                if json_match:
                    try:
                        result_dict = json.loads(json_match.group())
                        logger.info("Successfully parsed enhanced JSON from AIMessage")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse enhanced JSON, extracting fields separately")
                        # Extract each field separately
                        result_dict = self._extract_fields_from_text(message_content)
                else:
                    logger.warning("No JSON found in enhanced AIMessage content")
                    result_dict = self._extract_fields_from_text(message_content)
            
            elif isinstance(raw_result, str):
                # Handle string responses
                json_start = raw_result.find('{')
                json_end = raw_result.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = raw_result[json_start:json_end]
                    result_dict = json.loads(json_str)
                else:
                    result_dict = self._extract_fields_from_text(raw_result)
            
            elif isinstance(raw_result, dict):
                result_dict = raw_result
            else:
                result_dict = self._extract_fields_from_text(str(raw_result))
            
            # Ensure all required fields are present with defaults
            enhanced_result = {
                "got_answer": self._normalize_yes_no(result_dict.get("got_answer", "no")),
                "source_mentioned": self._normalize_yes_no(result_dict.get("source_mentioned", "no")),
                "answer_grounded": self._normalize_yes_no(result_dict.get("answer_grounded", "no")),
                "relevance_score": self._normalize_relevance(result_dict.get("relevance_score", "low"))
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error parsing enhanced guardrail result: {e}")
            # Fallback to basic assessment
            got_answer = "no" if "could not process document" in answer.lower() or "error" in answer.lower() else "yes"
            source_mentioned = "yes" if "page" in answer.lower() or "source" in answer.lower() else "no"
            
            return {
                "got_answer": got_answer,
                "source_mentioned": source_mentioned,
                "answer_grounded": "no",  # Conservative default
                "relevance_score": "low"  # Conservative default
            }
    
    def _extract_fields_from_text(self, text):
        """Extract fields from text using regex patterns"""
        import re
        
        result_dict = {}
        
        # Extract each field
        patterns = {
            "got_answer": r'"got_answer"\s*:\s*"([^"]*)"',
            "source_mentioned": r'"source_mentioned"\s*:\s*"([^"]*)"', 
            "answer_grounded": r'"answer_grounded"\s*:\s*"([^"]*)"',
            "relevance_score": r'"relevance_score"\s*:\s*"([^"]*)"'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                result_dict[field] = match.group(1)
        
        return result_dict
    
    def _normalize_yes_no(self, value):
        """Normalize a value to 'yes' or 'no'"""
        if not value:
            return "no"
        
        value_str = str(value).lower().strip()
        if value_str in ["1", "true", "yes", "y"]:
            return "yes"
        else:
            return "no"
    
    def _normalize_relevance(self, value):
        """Normalize relevance score to 'high', 'medium', or 'low'"""
        if not value:
            return "low"
        
        value_str = str(value).lower().strip()
        if value_str in ["high", "h"]:
            return "high"
        elif value_str in ["medium", "med", "m"]:
            return "medium"
        else:
            return "low"
    
    def _parse_guardrail_result(self, raw_result, answer):
        """Parse basic guardrail result (existing implementation)"""
        
        try:
            # Try multiple parsing approaches
            if isinstance(raw_result, str):
                # Clean the string to extract just the JSON part
                cleaned_result = raw_result.strip()
                json_start = cleaned_result.find('{')
                json_end = cleaned_result.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = cleaned_result[json_start:json_end]
                    result_dict = json.loads(json_str)
                else:
                    # If no JSON delimiters found, try the whole string
                    result_dict = json.loads(cleaned_result)
            elif isinstance(raw_result, dict):
                # Already a dictionary
                result_dict = raw_result
            else:
                # Try to access as object attributes
                result_dict = {
                    "got_answer": getattr(raw_result, "got_answer", "no"),
                    "source_mentioned": getattr(raw_result, "source_mentioned", "no")
                }
                
            # Normalize values to ensure "yes" or "no"
            for key in ["got_answer", "source_mentioned"]:
                if key not in result_dict:
                    result_dict[key] = "no"
                else:
                    # Normalize to lowercase yes/no
                    value = str(result_dict[key]).lower().strip()
                    if value in ["1", "true", "yes", "y"]:
                        result_dict[key] = "yes"
                    else:
                        result_dict[key] = "no"
                        
            logger.info(f"Successfully parsed basic guardrail result: {result_dict}")
            return result_dict
                
        except Exception as parsing_error:
            logger.warning(f"Basic parsing failed: {parsing_error}, trying regex fallback")
            
            # Attempt regex fallback
            import re
            json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
            matches = re.findall(json_pattern, str(raw_result), re.DOTALL)
            
            if matches:
                for potential_json in matches:
                    try:
                        result_dict = json.loads(potential_json)
                        if "got_answer" in result_dict or "source_mentioned" in result_dict:
                            # Normalize values
                            for key in ["got_answer", "source_mentioned"]:
                                if key not in result_dict:
                                    result_dict[key] = "no"
                                else:
                                    value = str(result_dict[key]).lower().strip()
                                    result_dict[key] = "yes" if value in ["1", "true", "yes", "y"] else "no"
                                    
                            logger.info(f"Regex extraction successful: {result_dict}")
                            return result_dict
                    except json.JSONDecodeError:
                        continue
                        
            # Text analysis fallback
            got_answer = "yes" if any(x in str(raw_result).lower() for x in [
                "provided a substantive answer",
                "addresses the query", 
                "answered the question",
                "got_answer\": \"yes",
                "got_answer\":\"yes"
            ]) else "no"
            
            source_mentioned = "yes" if any(x in str(raw_result).lower() for x in [
                "includes references",
                "cites specific sources",
                "page numbers",
                "mentions sources",
                "source_mentioned\": \"yes",
                "source_mentioned\":\"yes"
            ]) else "no"
            
            logger.info(f"Text analysis extraction: got_answer={got_answer}, source_mentioned={source_mentioned}")
            return {"got_answer": got_answer, "source_mentioned": source_mentioned}
    
    def modify_query(self, original_query: str, answer: str, issue_type: str) -> str:
        """Create a modified query based on issues with the original answer"""
        if issue_type == "answer":
            prompt = f"""
            Original query was: "{original_query}"
            
            However, it could not find the answer. You need to rephrase the question to be clearer
            and more specific to help the model find the relevant information.
            
            Provide only the rephrased question with no additional text or explanation.
            """
        elif issue_type == "source":
            prompt = f"""
            Original query was: "{original_query}"
            
            The answer was received: "{answer}"
            
            However, it did not include page numbers or section references. Create a follow-up question
            that specifically asks for the page numbers or sections where this information can be found.
            
            Provide only the follow-up question with no additional text or explanation.
            """
        elif issue_type == "grounding":
            prompt = f"""
            Original query was: "{original_query}"
            
            The answer was: "{answer}"
            
            However, the answer appears to not be well-grounded in the source material or may contain
            information not supported by the documents. Create a more specific question that asks for
            information that can be directly found and quoted from the source documents.
            
            Remember to use {original_query} as the base question. And only add something to it, when you think it will help ground the answer.
            Do not completely change the question.
            
            Provide only the modified question with no additional text or explanation.
            """
        else:
            return original_query
            
        try:
            # Get the appropriate LLM
            llm = self.get_guardrail_llm()
            
            modified_query = llm.invoke(prompt)
            logger.info(f"Modified query for {issue_type}: {modified_query}")
            return modified_query.strip()
        except Exception as e:
            logger.error(f"Error modifying query: {e}")
            return original_query

class ScoringAgent:
    """Handles scoring of answers based on defined criteria"""
    
    def __init__(self, config: CGSConfig):
        self.config = config
        
        # Initialize LLMs based on configuration
        try:
            # Always initialize Gemini as it's needed for various functions
            self.gemini = ChatGoogleGenerativeAI(model=config.gemini_model)
            logger.info(f"Initialized Gemini client with model {config.gemini_model}")
            
            # Initialize Ollama if that's the selected provider
            if config.model_provider == "ollama":
                try:
                    self.ollama = OllamaLLM(model=config.model_to_use)
                    logger.info(f"Initialized Ollama with model {config.model_to_use}")
                except Exception as e:
                    logger.error(f"Error initializing Ollama: {e}")
                    logger.warning("Falling back to Gemini for all operations")
                    self.config.model_provider = "gemini"
        except Exception as e:
            logger.error(f"Error initializing LLMs: {e}")
    
    def get_scoring_llm(self):
        """Get the appropriate LLM for scoring based on configuration"""
        if self.config.model_provider == "ollama" and hasattr(self, 'ollama'):
            return self.ollama
        else:
            return self.gemini
    
    def postprocess_content(self, question_no: int, content: str) -> str:
        """Apply special post-processing to certain questions"""
        # Only question 12 has special processing in the original code
        if question_no != 12:
            return content
            
        # Special processing for question 12 (royalty transactions)
        prompt = ChatPromptTemplate.from_template(
            """
            This context has information about related party transactions.
            
            Please analyze the content and:
            1. Identify and filter all transactions related to Royalty payments
            2. Calculate the total amount of these Royalty transactions
            3. Compare this total to the company's profit, which should be mentioned in the content
            4. If royalty is not mentioned, assume it is 0
            
            Context: {content}
            
            Provide a detailed analysis with specific amounts and percentages.
            """
        )
        
        # Create chain
        chain = (
            {"content": lambda x: x}
            | prompt
            | self.gemini
            | StrOutputParser()
        )
        
        try:
            processed_content = chain.invoke(content)
            logger.info(f"Post-processed content for question {question_no}")
            return processed_content
        except Exception as e:
            logger.error(f"Error in content post-processing: {e}")
            return content
    
    # def score_answer(self, scoring_criteria: str, content: str) -> Dict[str, Union[int, str]]:
    #     """Score the content based on provided criteria with extra debugging for zero scores"""
        
    #     # Check for empty inputs
    #     if not content or not content.strip():
    #         logger.warning("Empty content provided for scoring")
    #         return {
    #             "score": 0,
    #             "justification": "No content provided for scoring. Please ensure questions were processed successfully."
    #         }
        
    #     if not scoring_criteria or not scoring_criteria.strip():
    #         logger.warning("Empty scoring criteria provided")
    #         return {
    #             "score": 0,
    #             "justification": "No scoring criteria provided. Please check your scoring_creteria.csv file."
    #         }
        
    #     # Get the appropriate LLM
    #     llm = self.get_scoring_llm()
    #     using_gemini = self.config.model_provider == "gemini"
        
    #     # Log which LLM we're using
    #     logger.info(f"Scoring using provider: {self.config.model_provider}")
    #     if using_gemini:
    #         logger.info(f"Using Gemini model: {self.config.gemini_model}")
    #     else:
    #         logger.info(f"Using Ollama model: {self.config.model_to_use}")
        
    #     # Create a modified prompt specifically designed for better Gemini performance
    #     prompt_template = """
    #     You are a corporate governance scoring expert. Your task is to evaluate corporate governance based on specific criteria and content.
        
    #     SCORING CRITERIA:
    #     {scoring_criteria}
        
    #     CONTENT TO EVALUATE:
    #     {content}
        
    #     SCORING RULES:
    #     - Score from 0 to 10, where 0 means criteria not met at all, and 10 means fully met
    #     - If information is missing, score should be 0
    #     - Include page numbers and document references in your justification where possible
    #     - Be objective and thorough in your evaluation
        
    #     YOUR RESPONSE MUST BE A VALID JSON OBJECT WITH EXACTLY THIS FORMAT:
    #     {{"score": <integer_0_to_10>, "justification": "<detailed_explanation>"}}
        
    #     The score MUST be an integer number between 0 and 10.
    #     Do not include any text before or after the JSON object.
    #     """
        
    #     prompt = ChatPromptTemplate.from_template(prompt_template)
        
    #     # Create the chain
    #     chain = (
    #         {"scoring_criteria": lambda x: x[0], "content": lambda x: x[1]}
    #         | prompt
    #         | llm
    #     )
        
    #     try:
    #         # Run the chain
    #         start_time = time.time()
    #         raw_result = chain.invoke([scoring_criteria, content])
    #         end_time = time.time()
    #         logger.info(f"Model response time: {end_time - start_time:.2f} seconds")
            
    #         # Log raw result type and sample
    #         result_type = type(raw_result).__name__
    #         result_sample = str(raw_result)[:500] + ("..." if len(str(raw_result)) > 500 else "")
    #         logger.info(f"Raw result type: {result_type}")
    #         logger.info(f"Raw result sample: {result_sample}")
            
    #         # Parse the result
    #         try:
    #             # Try to find JSON in the response
    #             if isinstance(raw_result, str):
    #                 cleaned_result = raw_result.strip()
    #                 # Look for JSON object
    #                 json_start = cleaned_result.find('{')
    #                 json_end = cleaned_result.rfind('}') + 1
                    
    #                 if json_start >= 0 and json_end > json_start:
    #                     # Extract the JSON part
    #                     json_str = cleaned_result[json_start:json_end]
    #                     result_dict = json.loads(json_str)
    #                     logger.info(f"Successfully extracted and parsed JSON from string response")
    #                 else:
    #                     # Try to parse the whole string
    #                     result_dict = json.loads(cleaned_result)
    #                     logger.info(f"Parsed entire response as JSON")
    #             elif isinstance(raw_result, dict):
    #                 # Already a dictionary
    #                 result_dict = raw_result
    #                 logger.info(f"Response was already a dictionary")
    #             else:
    #                 # Try to access as object attributes
    #                 try:
    #                     result_dict = {
    #                         "score": getattr(raw_result, "score", 0),
    #                         "justification": getattr(raw_result, "justification", "No justification provided")
    #                     }
    #                     logger.info(f"Extracted score from object attributes")
    #                 except:
    #                     logger.warning(f"Could not extract attributes from {result_type} response")
    #                     result_dict = {"score": 0, "justification": f"Could not parse response of type {result_type}"}
    #         except Exception as parse_error:
    #             logger.error(f"Error parsing result: {parse_error}")
    #             logger.error(f"Problem parsing: {result_sample}")
                
    #             # Last-resort parsing to extract a score
    #             try:
    #                 # Try to find a score using regex
    #                 import re
    #                 score_pattern = r'"score"\s*:\s*(\d+)'
    #                 score_match = re.search(score_pattern, str(raw_result))
                    
    #                 if score_match:
    #                     score = int(score_match.group(1))
    #                     logger.info(f"Extracted score {score} using regex")
                        
    #                     # Try to extract justification
    #                     justification_pattern = r'"justification"\s*:\s*"([^"]*)"'
    #                     justification_match = re.search(justification_pattern, str(raw_result))
    #                     justification = justification_match.group(1) if justification_match else "Justification extraction failed"
                        
    #                     result_dict = {"score": score, "justification": justification}
    #                 else:
    #                     # Give up and return a default
    #                     logger.warning("No score found in response, defaulting to 0")
    #                     result_dict = {"score": 0, "justification": f"Failed to parse response: {parse_error}"}
    #             except:
    #                 result_dict = {"score": 0, "justification": f"Failed to parse response: {parse_error}"}
            
    #         # Process and validate the result
    #         if "score" not in result_dict:
    #             logger.warning("No score in parsed result, defaulting to 0")
    #             result_dict["score"] = 0
            
    #         if "justification" not in result_dict:
    #             logger.warning("No justification in parsed result")
    #             result_dict["justification"] = "No justification provided by the model"
            
    #         # Ensure score is an integer
    #         try:
    #             result_dict["score"] = int(float(result_dict["score"]))
    #         except:
    #             logger.warning(f"Could not convert score to integer: {result_dict.get('score')}")
    #             result_dict["score"] = 0
            
    #         # Ensure score is in range
    #         if result_dict["score"] < 0 or result_dict["score"] > 10:
    #             original_score = result_dict["score"]
    #             result_dict["score"] = max(0, min(10, result_dict["score"]))
    #             logger.warning(f"Clamped score from {original_score} to {result_dict['score']}")
            
    #         # Check for zero score and log details
    #         if result_dict["score"] == 0:
    #             logger.warning("Zero score detected, this might indicate a problem")
    #             logger.warning(f"Justification for zero score: {result_dict.get('justification', 'None provided')[:200]}...")
                
    #             # Enhanced justification for zero scores
    #             if "no information" in result_dict.get("justification", "").lower() or "information is missing" in result_dict.get("justification", "").lower():
    #                 logger.info("Zero score appears to be due to missing information")
    #             elif "criteria not met" in result_dict.get("justification", "").lower():
    #                 logger.info("Zero score appears to be due to criteria not being met")
    #             else:
    #                 logger.warning("Zero score reason is unclear from justification")
            
    #         return result_dict
        
    #     except Exception as e:
    #         logger.error(f"Error in scoring process: {e}")
    #         import traceback
    #         logger.error(f"Error traceback: {traceback.format_exc()}")
            
    #         return {
    #             "score": 0,
    #             "justification": f"Error in scoring process: {str(e)}"
    #         }
    
    def score_answer(self, scoring_criteria: str, content: str) -> Dict[str, Union[int, str]]:
        """Score the content based on provided criteria with improved handling of AIMessage type"""
        
        # Check for empty inputs
        if not content or not content.strip():
            logger.warning("Empty content provided for scoring")
            return {
                "score": 0,
                "justification": "No content provided for scoring. Please ensure questions were processed successfully."
            }
        
        if not scoring_criteria or not scoring_criteria.strip():
            logger.warning("Empty scoring criteria provided")
            return {
                "score": 0,
                "justification": "No scoring criteria provided. Please check your scoring_creteria.csv file."
            }
        
        # Get the appropriate LLM
        llm = self.get_scoring_llm()
        using_gemini = self.config.model_provider == "gemini"
        
        # Log which LLM we're using
        logger.info(f"Scoring using provider: {self.config.model_provider}")
        if using_gemini:
            logger.info(f"Using Gemini model: {self.config.gemini_model}")
        else:
            logger.info(f"Using Ollama model: {self.config.model_to_use}")
        
        # Create a modified prompt specifically designed for better Gemini performance
        prompt_template = """
        You are a corporate governance scoring expert. Your task is to evaluate corporate governance based on specific criteria and content.
        
        SCORING CRITERIA:
        {scoring_criteria}
        
        CONTENT TO EVALUATE:
        {content}
        
        SCORING RULES:
        - Score from 0 to 10, where 0 means criteria not met at all, and 10 means fully met
        - If information is missing, score should be 0
        - Include page numbers and document references in your justification where possible
        - Be objective and thorough in your evaluation
        
        YOUR RESPONSE MUST BE A VALID JSON OBJECT WITH EXACTLY THIS FORMAT:
        {{"score": <integer_0_to_10>, "justification": "<detailed_explanation>"}}
        
        The score MUST be an integer number between 0 and 10.
        Do not include any text before or after the JSON object.
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the chain
        chain = (
            {"scoring_criteria": lambda x: x[0], "content": lambda x: x[1]}
            | prompt
            | llm
        )
        
        try:
            # Run the chain
            start_time = time.time()
            raw_result = chain.invoke([scoring_criteria, content])
            end_time = time.time()
            logger.info(f"Model response time: {end_time - start_time:.2f} seconds")
            
            # Log raw result type and sample
            result_type = type(raw_result).__name__
            result_sample = str(raw_result)[:500] + ("..." if len(str(raw_result)) > 500 else "")
            logger.info(f"Raw result type: {result_type}")
            logger.info(f"Raw result sample: {result_sample}")
            
            # Special handling for AIMessage type (common with Gemini)
            if hasattr(raw_result, 'content') and isinstance(raw_result.content, str):
                # Extract content from AIMessage
                message_content = raw_result.content
                
                # Log the FULL message content from Gemini
                logger.info(f"=== FULL GEMINI RESPONSE ===")
                logger.info(message_content)
                logger.info(f"=== END GEMINI RESPONSE ===")
                
                # Look for JSON in the content - often in markdown code blocks with ```json
                import re
                # Try to extract JSON from markdown code blocks
                json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
                json_match = re.search(json_block_pattern, message_content, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1).strip()
                    logger.info(f"Extracted JSON from markdown block: {json_str[:100]}...")
                    try:
                        result_dict = json.loads(json_str)
                        logger.info("Successfully parsed JSON from markdown block")
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON decode error from markdown block: {je}")
                        # Fall back to normal JSON extraction
                        json_pattern = r'\{.*\}'
                        json_match = re.search(json_pattern, message_content, re.DOTALL)
                        if json_match:
                            try:
                                result_dict = json.loads(json_match.group())
                                logger.info("Successfully parsed JSON using regex fallback")
                            except json.JSONDecodeError:
                                logger.error("Failed to parse JSON with regex fallback")
                                # Extract score and justification separately
                                score_match = re.search(r'"score"\s*:\s*(\d+)', message_content)
                                justification_match = re.search(r'"justification"\s*:\s*"([^"]*)"', message_content)
                                
                                score = int(score_match.group(1)) if score_match else 0
                                justification = justification_match.group(1) if justification_match else "No justification found in response"
                                
                                result_dict = {
                                    "score": score,
                                    "justification": justification
                                }
                else:
                    # Try direct JSON extraction
                    json_pattern = r'\{.*\}'
                    json_match = re.search(json_pattern, message_content, re.DOTALL)
                    if json_match:
                        try:
                            result_dict = json.loads(json_match.group())
                            logger.info("Successfully parsed JSON using regex")
                        except json.JSONDecodeError:
                            # Extract score and justification separately
                            score_match = re.search(r'"score"\s*:\s*(\d+)', message_content)
                            justification_match = re.search(r'"justification"\s*:\s*"(.*?)"', message_content, re.DOTALL)
                            
                            score = int(score_match.group(1)) if score_match else 0
                            justification = justification_match.group(1) if justification_match else "No justification found in response"
                            
                            result_dict = {
                                "score": score,
                                "justification": justification
                            }
                    else:
                        logger.warning("No JSON found in AIMessage content")
                        # Create a basic dictionary with entire content as justification
                        result_dict = {
                            "score": 0,
                            "justification": "Could not extract score. Full response: " + message_content[:500]
                        }
            else:
                # Parse the result for other types
                try:
                    # Try to find JSON in the response
                    if isinstance(raw_result, str):
                        cleaned_result = raw_result.strip()
                        # Look for JSON object
                        json_start = cleaned_result.find('{')
                        json_end = cleaned_result.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            # Extract the JSON part
                            json_str = cleaned_result[json_start:json_end]
                            result_dict = json.loads(json_str)
                            logger.info(f"Successfully extracted and parsed JSON from string response")
                        else:
                            # Try to parse the whole string
                            result_dict = json.loads(cleaned_result)
                            logger.info(f"Parsed entire response as JSON")
                    elif isinstance(raw_result, dict):
                        # Already a dictionary
                        result_dict = raw_result
                        logger.info(f"Response was already a dictionary")
                    else:
                        # Try to access as object attributes
                        try:
                            result_dict = {
                                "score": getattr(raw_result, "score", 0),
                                "justification": getattr(raw_result, "justification", "No justification provided")
                            }
                            logger.info(f"Extracted score from object attributes")
                        except:
                            logger.warning(f"Could not extract attributes from {result_type} response")
                            result_dict = {"score": 0, "justification": f"Could not parse response of type {result_type}"}
                except Exception as parse_error:
                    logger.error(f"Error parsing result: {parse_error}")
                    logger.error(f"Problem parsing: {result_sample}")
                    
                    # Last-resort parsing to extract a score
                    try:
                        # Try to find a score using regex
                        import re
                        score_pattern = r'"score"\s*:\s*(\d+)'
                        score_match = re.search(score_pattern, str(raw_result))
                        
                        if score_match:
                            score = int(score_match.group(1))
                            logger.info(f"Extracted score {score} using regex")
                            
                            # Try to extract justification
                            justification_pattern = r'"justification"\s*:\s*"([^"]*)"'
                            justification_match = re.search(justification_pattern, str(raw_result))
                            justification = justification_match.group(1) if justification_match else "Justification extraction failed"
                            
                            result_dict = {"score": score, "justification": justification}
                        else:
                            # Give up and return a default
                            logger.warning("No score found in response, defaulting to 0")
                            result_dict = {"score": 0, "justification": f"Failed to parse response: {parse_error}"}
                    except:
                        result_dict = {"score": 0, "justification": f"Failed to parse response: {parse_error}"}
            
            # Process and validate the result
            if "score" not in result_dict:
                logger.warning("No score in parsed result, defaulting to 0")
                result_dict["score"] = 0
            
            if "justification" not in result_dict:
                logger.warning("No justification in parsed result")
                result_dict["justification"] = "No justification provided by the model"
            
            # Ensure score is an integer
            try:
                result_dict["score"] = int(float(result_dict["score"]))
            except:
                logger.warning(f"Could not convert score to integer: {result_dict.get('score')}")
                result_dict["score"] = 0
            
            # Ensure score is in range
            if result_dict["score"] < 0 or result_dict["score"] > 10:
                original_score = result_dict["score"]
                result_dict["score"] = max(0, min(10, result_dict["score"]))
                logger.warning(f"Clamped score from {original_score} to {result_dict['score']}")
            
            # Check for zero score and log details
            if result_dict["score"] == 0:
                logger.warning("Zero score detected, this might indicate a problem")
                justification_preview = result_dict.get("justification", "None provided")[:200]
                logger.warning(f"Justification for zero score: {justification_preview}...")
                
                # Enhanced justification for zero scores
                if "no information" in result_dict.get("justification", "").lower() or "information is missing" in result_dict.get("justification", "").lower():
                    logger.info("Zero score appears to be due to missing information")
                elif "criteria not met" in result_dict.get("justification", "").lower():
                    logger.info("Zero score appears to be due to criteria not being met")
                else:
                    logger.warning("Zero score reason is unclear from justification")
            
            return result_dict
        
        except Exception as e:
            logger.error(f"Error in scoring process: {e}")
            import traceback
            logger.error(f"Error traceback: {traceback.format_exc()}")
            
            return {
                "score": 0,
                "justification": f"Error in scoring process: {str(e)}"
            }
        
class CorporateGovernanceAgent:
    """Main agent that orchestrates the entire workflow"""
    
    def __init__(self, company_sym: str, base_path: str = None, config: CGSConfig = None):
        """Initialize the agent"""
        # Initialize configuration
        self.config = config if config else CGSConfig(company_sym, base_path)
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.query_engine = QueryEngine(self.config, self.document_processor)
        self.guardrail_agent = GuardrailAgent(self.config)
        self.scoring_agent = ScoringAgent(self.config)
        
        # Load document source map if available
        self.source_map = {}
        self.load_source_map()
        
        
    
    def load_source_map(self):
        """Load existing source map if available"""
        source_map_path = os.path.join(self.config.static_path, 'source_path_map.csv')
        if os.path.exists(source_map_path):
            df = pd.read_csv(source_map_path)
            self.source_map = dict(zip(df['source'], df['path']))
            logger.info(f"Loaded existing source map with {len(self.source_map)} entries")
    
    def setup(self):
        """Set up the agent by downloading documents and creating source maps"""
        logger.info(f"Setting up agent for company {self.config.company_sym}")
        
        # Download PDFs from URLs if needed
        downloaded_sources = self.document_processor.download_pdfs_from_csv()
        if downloaded_sources:
            logger.info(f"Downloaded {len(downloaded_sources)} PDFs")
            self.source_map.update(downloaded_sources)
        
        # Split large PDFs if needed
        self.document_processor.split_large_pdfs()
        
        # Create or update source map
        self.source_map = self.document_processor.create_source_map()
        logger.info(f"Source map created with {len(self.source_map)} entries")
        
        return self
    
    def process_questions(self, load_all_fresh=False, sr_no_list=None):
        """Process questions from prompts.csv and get answers with enhanced guardrail validation"""
        logger.info("Processing questions from prompts.csv")
        
        # Load prompts
        prompts_path = os.path.join(self.config.parent_path, 'prompts.csv')
        if not os.path.exists(prompts_path):
            logger.error(f"Prompts file not found: {prompts_path}")
            return
            
        prompts_df = pd.read_csv(prompts_path)
        
        # Load existing results if available
        results_path = os.path.join(self.config.results_path, 'prompts_result.csv')
        
        if not load_all_fresh and os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            # Determine which questions have already been processed
            processed_sr_nos = results_df['sr_no'].unique()
            all_sr_nos = prompts_df['sr_no'].unique()
            
            # Filter to only process requested or unprocessed questions
            if sr_no_list:
                sr_nos_to_process = [x for x in all_sr_nos if x in sr_no_list]
            else:
                sr_nos_to_process = [x for x in all_sr_nos if x not in processed_sr_nos]
                
            # Filter prompts
            prompts_df = prompts_df[prompts_df['sr_no'].isin(sr_nos_to_process)]
            
            # Keep results for questions not being reprocessed
            results_df = results_df[~results_df['sr_no'].isin(sr_nos_to_process)]
        else:
            # Process only specified questions if requested
            if sr_no_list:
                prompts_df = prompts_df[prompts_df['sr_no'].isin(sr_no_list)]
            
            # Start with empty results
            results_df = pd.DataFrame(columns=['run_time_stamp', 'sr_no', 'cat', 'que_no', 'source', 'message', 'result'])
        
        # Process each question
        for _, row in prompts_df.iterrows():
            message = row['message']
            disp_msg = row.get('disp_message', message)
            que_no = row['que_no']
            cat = row['cat']
            sr_no = row['sr_no']
            source_filter = row['source']
            
            logger.info(f"Processing question {que_no} (sr_no: {sr_no}): {disp_msg}")
            
            # Determine which sources to use
            all_sources = []
            if source_filter == "ALL":
                all_sources = list(self.source_map.values())
            else:
                # Find sources that start with the filter
                all_sources = [v for k, v in self.source_map.items() if k.startswith(source_filter)]
            
            if not all_sources:
                logger.warning(f"No sources found for filter: {source_filter}")
                continue
                
            logger.info(f"Using {len(all_sources)} sources for question {que_no}")
            
            # Process each source
            for source_path in all_sources:
                logger.info(f"Processing source: {source_path}")
                
                # Get top chunks for guardrail validation (only if not using direct method)
                top_chunks = None
                if self.config.retrieval_method.lower() != "direct":
                    try:
                        # Get the retriever for this document
                        retriever = self.query_engine.document_processor.create_retriever(source_path)
                        if retriever:
                            # Get relevant chunks using the query
                            try:
                                top_chunks = retriever.invoke(message)
                            except (AttributeError, TypeError):
                                # Fall back to legacy method
                                try:
                                    top_chunks = retriever.get_relevant_documents(message)
                                except Exception as e:
                                    logger.warning(f"Could not retrieve chunks for guardrail validation: {e}")
                                    top_chunks = None
                            
                            # Limit chunks for guardrail validation (to avoid token limits)
                            if top_chunks and len(top_chunks) > 50:
                                top_chunks = top_chunks[:50]
                                
                            logger.info(f"Retrieved {len(top_chunks) if top_chunks else 0} chunks for guardrail validation")
                        else:
                            logger.warning(f"Could not create retriever for {source_path}")
                    except Exception as e:
                        logger.warning(f"Error getting chunks for guardrail validation: {e}")
                        top_chunks = None
                
                # Query the document
                result = self.query_engine.query_document(source_path, message)
                
                if not result:
                    logger.warning(f"No result obtained for {source_path}")
                    result = "Could not process document."
                
                # Apply enhanced guardrails with chunk context
                try:
                    guardrail_result = self.guardrail_agent.verify_answer_quality(
                        message, result, top_chunks=top_chunks
                    )
                    
                    # Convert to string values for consistency
                    if guardrail_result and isinstance(guardrail_result, dict):
                        got_answer = guardrail_result.get('got_answer', 'no')
                        source_mentioned = guardrail_result.get('source_mentioned', 'no')
                        answer_grounded = guardrail_result.get('answer_grounded', 'no')
                        relevance_score = guardrail_result.get('relevance_score', 'low')
                        
                        # Log enhanced guardrail results
                        logger.info(f"Enhanced guardrail results - Answer: {got_answer}, Sources: {source_mentioned}, "
                                f"Grounded: {answer_grounded}, Relevance: {relevance_score}")
                    else:
                        # Handle case where guardrail_result is an object with attributes
                        try:
                            got_answer = getattr(guardrail_result, 'got_answer', 'no')
                            source_mentioned = getattr(guardrail_result, 'source_mentioned', 'no')
                            answer_grounded = getattr(guardrail_result, 'answer_grounded', 'no')
                            relevance_score = getattr(guardrail_result, 'relevance_score', 'low')
                        except:
                            got_answer = 'no'
                            source_mentioned = 'no'
                            answer_grounded = 'no'
                            relevance_score = 'low'
                except Exception as e:
                    logger.error(f"Error in enhanced guardrail verification: {e}")
                    # Fallback to simple checks if guardrail fails
                    got_answer = "no" if "could not process document" in result.lower() or "error" in result.lower() else "yes"
                    source_mentioned = "yes" if "page" in result.lower() or "source" in result.lower() else "no"
                    answer_grounded = "no"  # Conservative default
                    relevance_score = "low"  # Conservative default
                
                # Try to improve answer based on enhanced guardrail feedback
                improvement_needed = False
                
                # Check if answer needs improvement (prioritize grounding issues)
                if answer_grounded == 'no' and got_answer == 'yes':
                    try:
                        # Address grounding issues first
                        modified_query = self.guardrail_agent.modify_query(message, result, "grounding")
                        logger.info(f"Modified query for better grounding: {modified_query}")
                        
                        # Query again with modified query
                        result_attempt2 = self.query_engine.query_document(source_path, modified_query)
                        if result_attempt2:
                            result = result + "\n\n[Follow-up for better grounding]: " + result_attempt2
                            improvement_needed = True
                    except Exception as e:
                        logger.error(f"Error in follow-up query for better grounding: {e}")
                
                # Try to improve answer if needed
                elif got_answer == 'no':
                    try:
                        # Modify the query to try to get a better answer
                        modified_query = self.guardrail_agent.modify_query(message, result, "answer")
                        logger.info(f"Modified query for better answer: {modified_query}")
                        
                        # Query again with modified query
                        result_attempt2 = self.query_engine.query_document(source_path, modified_query)
                        if result_attempt2:
                            result = result + "\n\n[Follow-up for better answer]: " + result_attempt2
                            improvement_needed = True
                    except Exception as e:
                        logger.error(f"Error in follow-up query for better answer: {e}")
                
                # Try to get source references if missing
                elif source_mentioned == 'no' and not improvement_needed:
                    try:
                        # Modify the query to try to get source references
                        modified_query = self.guardrail_agent.modify_query(message, result, "source")
                        logger.info(f"Modified query for source references: {modified_query}")
                        
                        # Query again with modified query
                        result_attempt2 = self.query_engine.query_document(source_path, modified_query)
                        if result_attempt2:
                            result = result + "\n\n[Follow-up for source references]: " + result_attempt2
                    except Exception as e:
                        logger.error(f"Error in follow-up query for source references: {e}")
                
                # Save result with enhanced metadata
                run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Create enhanced result entry with guardrail metadata
                enhanced_result = result
                if 'answer_grounded' in locals() and 'relevance_score' in locals():
                    # Add guardrail assessment as metadata in the result
                    guardrail_summary = f"\n\n[Guardrail Assessment - Grounded: {answer_grounded}, Relevance: {relevance_score}]"
                    enhanced_result = result + guardrail_summary
                    print(f"Enhanced result with guardrail summary: {enhanced_result[:200]}...")  # Log first 200 chars
                
                new_row = pd.DataFrame({
                    'run_time_stamp': [run_time_stamp],
                    'sr_no': [sr_no],
                    'cat': [cat],
                    'que_no': [que_no],
                    'source': [source_filter],
                    'message': [message],
                    'result': [enhanced_result]
                })
                
                # Append to results
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                # Save incrementally
                try:
                    results_df.to_csv(results_path, index=False)
                    logger.info(f"Saved result for question {que_no}, source {os.path.basename(source_path)}, saved at {results_path}")
                except Exception as e:
                    logger.error(f"Error saving results to CSV: {e}")
        
        logger.info("Completed processing all questions with enhanced guardrail validation")
        return results_df
    
           
    def score_topic(self, topic_no):
        """Score a specific topic with enhanced logging including full model responses"""
        try:
            logger.info(f"Scoring topic {topic_no}")
            
            # Get the current model provider for logging
            model_provider = "unknown"
            if hasattr(self.config, 'model_provider'):
                model_provider = self.config.model_provider
            
            # Add detailed logging before scoring
            logger.info(f"Using provider: {model_provider}")
            
            # Get content for the topic - this is a critical part
            content, cat = self._get_topic_content(topic_no)
            
            # Log content length for debugging
            content_length = len(content) if content else 0
            logger.info(f"Topic {topic_no} content length: {content_length} characters")
            
            if not content or content_length < 10:  # Arbitrary small threshold
                logger.warning(f"Content for topic {topic_no} is empty or very short. This may cause scoring issues.")
            
            # Get scoring criteria
            criteria_path = os.path.join(self.config.parent_path, 'scoring_creteria.csv')
            if os.path.exists(criteria_path):
                try:
                    sc_df = pd.read_csv(criteria_path)
                    topic_criteria = sc_df[sc_df['topic_no'] == topic_no]
                    if topic_criteria.empty:
                        logger.error(f"No scoring criteria found for topic {topic_no}")
                        return None
                    scoring_criteria = topic_criteria['scoring_criteria'].iloc[0]
                    
                    # Log criteria length for debugging
                    criteria_length = len(scoring_criteria) if scoring_criteria else 0
                    logger.info(f"Topic {topic_no} criteria length: {criteria_length} characters")
                    
                    if not scoring_criteria or criteria_length < 10:  # Arbitrary small threshold
                        logger.warning(f"Scoring criteria for topic {topic_no} is empty or very short.")
                except Exception as e:
                    logger.error(f"Error reading scoring criteria: {e}")
                    return None
            else:
                logger.error(f"Scoring criteria file not found: {criteria_path}")
                return None
            
            # Apply any special post-processing
            if content and content.strip():
                content = self.scoring_agent.postprocess_content(topic_no, content)
            
            # Score the content - add more logging
            logger.info(f"Sending topic {topic_no} to model for scoring (provider: {model_provider})")
            
            # Log sample of content and criteria (first 200 chars)
            content_sample = content[:200] + "..." if content and len(content) > 200 else content
            criteria_sample = scoring_criteria[:200] + "..." if scoring_criteria and len(scoring_criteria) > 200 else scoring_criteria
            logger.info(f"Content sample: {content_sample}")
            logger.info(f"Criteria sample: {criteria_sample}")
            
            # Score the content
            score_result = self.scoring_agent.score_answer(scoring_criteria, content)
            
            # Save the score
            score = score_result.get('score', 0)
            justification = score_result.get('justification', 'No justification provided')
            logger.info(f"Score result: {score} with justification length: {len(justification)}")
            
            # Log the FULL justification (not just a sample)
            logger.info(f"=== FULL JUSTIFICATION ===")
            logger.info(justification)
            logger.info(f"=== END JUSTIFICATION ===")
            
            self._save_score(topic_no, cat, score, justification)
            
            return score_result
        except Exception as e:
            logger.error(f"Error scoring topic: {e}")
            import traceback
            logger.error(f"Scoring error traceback: {traceback.format_exc()}")
            return None
                
    def _get_topic_content(self, topic_no):
        """Get combined content for a topic from prompts_result.csv"""
        file_path = os.path.join(self.config.results_path, 'prompts_result.csv')
        
        if not os.path.exists(file_path):
            logger.error(f"Results file not found: {file_path}")
            return None, None
            
        df = pd.read_csv(file_path)
        
        # Filter for the topic
        filtered_df = df[df['que_no'] == topic_no]
        
        if filtered_df.empty:
            logger.warning(f"No results found for topic {topic_no}")
            return None, None
            
        # Get category
        cat = filtered_df['cat'].iloc[0]
        
        # Combine all results
        content = '\n\n'.join(filtered_df['result'].dropna().tolist())
        
        return content, cat
    
    def _save_score(self, topic_no, category, score, justification):
        """Save score to que_wise_scores.csv"""
        file_path = os.path.join(self.config.results_path, 'que_wise_scores.csv')
        
        # Create or load existing scores
        if os.path.exists(file_path):
            scores_df = pd.read_csv(file_path)
        else:
            scores_df = pd.DataFrame(columns=[
                'run_time_stamp', 'company', 'category', 'que_no', 'score', 'justification'
            ])
        
        # Add new score
        run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_row = pd.DataFrame({
            'run_time_stamp': [run_time_stamp],
            'company': [self.config.company_sym],
            'category': [category],
            'que_no': [topic_no],
            'score': [score],
            'justification': [justification]
        })
        
        # Append to scores
        scores_df = pd.concat([scores_df, new_row], ignore_index=True)
        
        # Save
        scores_df.to_csv(file_path, index=False)
        logger.info(f"Saved score for topic {topic_no}: {score}")
        
        # Update the final scores
        self._select_latest_scores()
        
        return scores_df
    
    def _select_latest_scores(self):
        """Select latest scores for each question and save to que_wise_scores_final.csv"""
        file_path = os.path.join(self.config.results_path, 'que_wise_scores.csv')
        
        if not os.path.exists(file_path):
            logger.warning("No scores file found")
            return
            
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime for sorting
        df['run_time_stamp'] = pd.to_datetime(df['run_time_stamp'])
        
        # Select latest score for each question
        latest_df = df.loc[df.groupby('que_no')['run_time_stamp'].idxmax()]
        
        # Reset index
        latest_df = latest_df.reset_index(drop=True)
        
        # Save to final file
        final_path = os.path.join(self.config.results_path, 'que_wise_scores_final.csv')
        latest_df.to_csv(final_path, index=False)
        logger.info(f"Updated final scores with {len(latest_df)} entries")
        
        return latest_df
    
    def score_category(self, category_num):
        """Score all topics in a specific category"""
        logger.info(f"Scoring category {category_num}")
        
        if category_num == 1:
            # Rights and equitable treatment of shareholders
            self.score_topic(4)   # Board permanency
            self.score_topic(7)   # AGM delays
            self.score_topic(10)  # Conflict of interest
            self.score_topic(12)  # Royalty in related party transactions
            self.score_topic(16)  # Violation of minority shareholders rights
        
        elif category_num == 2:
            # Role of stakeholders
            self.score_topic(18)  # Health, safety, and welfare
            self.score_topic(19)  # Sexual harassment policy
            self.score_topic(20)  # Supplier/vendor selection/management
            self.score_topic(21)  # Delay in payment to stakeholders
            self.score_topic(22)  # Anti-corruption/Anti-bribery
            self.score_topic(23)  # CSR Spent
        
        elif category_num == 3:
            # Transparency and disclosure
            self.score_topic(28)  # Auditor's opinion
            self.score_topic(32)  # RPT policy
            self.score_topic(36)  # Shareholding pattern
            self.score_topic(37)  # Shareholding pattern board/KMP
            self.score_topic(38)  # Dividend distribution policy
            self.score_topic(44)  # Board qualification
            self.score_topic(45)  # Check for any fines
        
        elif category_num == 4:
            # Responsibility of the board
            self.score_topic(48)  # Attendance percentage in meetings
            self.score_topic(49)  # Nr times board met
            self.score_topic(51)  # Board expertise
            self.score_topic(52)  # Gender diversity on board
            self.score_topic(53)  # Gender diversity on workforce
            self.score_topic(54)  # Board independence
            self.score_topic(55)  # Committee checks
            self.score_topic(63)  # CEO compensation
    
    def score_all_categories(self):
        """Score all categories"""
        logger.info("Scoring all categories")
        
        for category in range(1, 5):
            self.score_category(category)
            
        logger.info("All categories scored")
    
    def aggregate_results(self):
        """Aggregate results from multiple companies"""
        logger.info("Aggregating results from all companies")
        
        parent_path = self.config.parent_path
        all_companies = [f.name for f in os.scandir(parent_path) if f.is_dir() and f.name != '95_all_results']
        
        # Create output directory
        os.makedirs(os.path.join(parent_path, '95_all_results'), exist_ok=True)
        
        # Aggregate prompt results
        all_prompt_results = pd.DataFrame()
        all_que_results = pd.DataFrame()
        
        for company in all_companies:
            # Aggregate prompt results
            prompt_result_path = os.path.join(parent_path, company, '96_results', 'prompts_result.csv')
            if os.path.exists(prompt_result_path):
                company_results = pd.read_csv(prompt_result_path)
                all_prompt_results = pd.concat([all_prompt_results, company_results], ignore_index=True)
            
            # Aggregate question scores
            que_result_path = os.path.join(parent_path, company, '96_results', 'que_wise_scores_final.csv')
            if os.path.exists(que_result_path):
                company_scores = pd.read_csv(que_result_path)
                all_que_results = pd.concat([all_que_results, company_scores], ignore_index=True)
        
        # Save aggregated results
        all_prompt_results.to_csv(os.path.join(parent_path, '95_all_results', 'all_promp_results.csv'), index=False)
        all_que_results.to_csv(os.path.join(parent_path, '95_all_results', 'all_que_results.csv'), index=False)
        
        logger.info(f"Aggregated results from {len(all_companies)} companies")
        return all_prompt_results, all_que_results
    
class GovernanceAgentCreator:
    """Creates a unified agent for corporate governance analysis"""
    
    def __init__(self, config: "CGSConfig"):
        self.config = config
        
    def create_master_agent(self, document_processor, query_engine, guardrail_agent, scoring_agent):
        """Create a master agent that combines all capabilities"""
        
        # Define tools for the agent
        tools = []
        
        # Document processing tools
        tools.append(
            StructuredTool.from_function(
                func=document_processor.split_large_pdfs,
                name="SplitLargePDFs",
                description="Split PDFs that are too large for processing into smaller chunks"
            )
        )
        
        # Query tools
        tools.append(
            StructuredTool.from_function(
                func=query_engine.query_document,
                name="QueryDocument",
                description="Query a document with a specific prompt"
            )
        )
        
        # Guardrail tools
        tools.append(
            StructuredTool.from_function(
                func=guardrail_agent.verify_answer_quality,
                name="VerifyAnswerQuality",
                description="Check if an answer is high quality and contains proper citations"
            )
        )
        
        # Scoring tools
        tools.append(
            StructuredTool.from_function(
                func=scoring_agent.score_answer,
                name="ScoreAnswer",
                description="Score content based on provided criteria"
            )
        )
        
        tools.append(
            StructuredTool.from_function(
                func=scoring_agent.postprocess_content,
                name="PostprocessContent",
                description="Apply special post-processing to content for specific questions"
            )
        )
        
        # Initialize LLM
        try:
            llm = OllamaLLM(model=self.config.model_to_use)
            logger.info(f"Initialized Ollama with model {self.config.model_to_use}")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                logger.info("Initialized Gemini as fallback")
            except Exception as e:
                logger.error(f"Error initializing Gemini: {e}")
                raise RuntimeError("Failed to initialize any LLM")
        
        # Create system prompt
        system_prompt = """
        You are an expert corporate governance analyzer and scorer.
        
        Your task is to analyze corporate governance documents (annual reports, financial statements, etc.)
        and score companies based on defined governance criteria.
        
        You have access to the following tools:
        - SplitLargePDFs: Split large PDFs into manageable chunks
        - QueryDocument: Ask questions about specific documents
        - VerifyAnswerQuality: Check if answers are high quality and properly cited
        - ScoreAnswer: Score content based on governance criteria
        - PostprocessContent: Apply special processing to certain types of content
        
        Follow a step-by-step approach:
        1. Understand the question or task
        2. Identify relevant documents to query
        3. Formulate specific queries to extract information
        4. Verify the quality of answers obtained
        5. Process and score the information as needed
        
        Always be thorough and accurate in your analysis.
        """
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(llm, tools, prompt)
        
        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        
        return agent_executor

# For advanced usage: Create a master agent that can handle complex tasks
def create_master_agent(company_sym):
    """Create a master agent for complex governance analysis tasks"""
    config = CGSConfig(company_sym)
    
    # Initialize components
    document_processor = DocumentProcessor(config)
    query_engine = QueryEngine(config, document_processor)
    guardrail_agent = GuardrailAgent(config)
    scoring_agent = ScoringAgent(config)
    
    # Create master agent
    agent_creator = GovernanceAgentCreator(config)
    master_agent = agent_creator.create_master_agent(
        document_processor, query_engine, guardrail_agent, scoring_agent
    )
    
    return master_agent


def aggregate_all_companies(base_parent_path):
    """Standalone function to aggregate results from all companies"""
    
    logger.info(f"Starting aggregation for all companies in: {base_parent_path}")
    
    try:
        # Validate base path exists
        if not os.path.exists(base_parent_path):
            logger.error(f"Base path does not exist: {base_parent_path}")
            return None, None
        
        # Get all company directories (exclude the results directory and any hidden folders)
        all_companies = []
        for item in os.scandir(base_parent_path):
            if (item.is_dir() and 
                not item.name.startswith('.') and 
                item.name != '95_all_results' and
                not item.name.startswith('__')):
                all_companies.append(item.name)
        
        if not all_companies:
            logger.warning(f"No company directories found in: {base_parent_path}")
            return None, None
        
        logger.info(f"Found {len(all_companies)} company directories: {all_companies}")
        
        # Create output directory
        output_dir = os.path.join(base_parent_path, '95_all_results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize DataFrames for aggregation
        all_prompt_results = pd.DataFrame()
        all_que_results = pd.DataFrame()
        
        # Track statistics
        companies_with_prompts = 0
        companies_with_scores = 0
        total_prompt_records = 0
        total_score_records = 0
        
        # Process each company
        for company in all_companies:
            logger.info(f"Processing company: {company}")
            
            # Aggregate prompt results (question-answer pairs)
            prompt_result_path = os.path.join(base_parent_path, company, '96_results', 'prompts_result.csv')
            if os.path.exists(prompt_result_path):
                try:
                    company_results = pd.read_csv(prompt_result_path)
                    
                    # Add company identifier if not already present
                    if 'company' not in company_results.columns:
                        company_results['company'] = company
                    
                    # Ensure consistent column order
                    expected_prompt_columns = ['run_time_stamp', 'company', 'sr_no', 'cat', 'que_no', 'source', 'message', 'result']
                    for col in expected_prompt_columns:
                        if col not in company_results.columns:
                            company_results[col] = None
                    
                    company_results = company_results[expected_prompt_columns]
                    all_prompt_results = pd.concat([all_prompt_results, company_results], ignore_index=True)
                    
                    companies_with_prompts += 1
                    total_prompt_records += len(company_results)
                    logger.info(f"  - Added {len(company_results)} prompt results from {company}")
                    
                except Exception as e:
                    logger.error(f"  - Error reading prompt results for {company}: {e}")
            else:
                logger.warning(f"  - No prompt results found for {company}: {prompt_result_path}")
            
            # Aggregate question scores (final scores)
            que_result_path = os.path.join(base_parent_path, company, '96_results', 'que_wise_scores_final.csv')
            if os.path.exists(que_result_path):
                try:
                    company_scores = pd.read_csv(que_result_path)
                    
                    # Add company identifier if not already present
                    if 'company' not in company_scores.columns:
                        company_scores['company'] = company
                    
                    # Ensure consistent column order
                    expected_score_columns = ['run_time_stamp', 'company', 'category', 'que_no', 'score', 'justification']
                    for col in expected_score_columns:
                        if col not in company_scores.columns:
                            company_scores[col] = None
                    
                    company_scores = company_scores[expected_score_columns]
                    all_que_results = pd.concat([all_que_results, company_scores], ignore_index=True)
                    
                    companies_with_scores += 1
                    total_score_records += len(company_scores)
                    logger.info(f"  - Added {len(company_scores)} score results from {company}")
                    
                except Exception as e:
                    logger.error(f"  - Error reading score results for {company}: {e}")
            else:
                logger.warning(f"  - No score results found for {company}: {que_result_path}")
        
        # Save aggregated results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save prompt results
        if not all_prompt_results.empty:
            prompt_output_path = os.path.join(output_dir, 'all_prompt_results.csv')
            prompt_backup_path = os.path.join(output_dir, f'all_prompt_results_backup_{timestamp}.csv')
            
            # Create backup if file exists
            if os.path.exists(prompt_output_path):
                try:
                    existing_df = pd.read_csv(prompt_output_path)
                    existing_df.to_csv(prompt_backup_path, index=False)
                    logger.info(f"Created backup of existing prompt results: {prompt_backup_path}")
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")
            
            # Save new aggregated results
            all_prompt_results.to_csv(prompt_output_path, index=False)
            logger.info(f"Saved aggregated prompt results to: {prompt_output_path}")
            logger.info(f"Total prompt records: {len(all_prompt_results)}")
        else:
            logger.warning("No prompt results to aggregate")
        
        # Save score results  
        if not all_que_results.empty:
            score_output_path = os.path.join(output_dir, 'all_question_scores.csv')
            score_backup_path = os.path.join(output_dir, f'all_question_scores_backup_{timestamp}.csv')
            
            # Create backup if file exists
            if os.path.exists(score_output_path):
                try:
                    existing_df = pd.read_csv(score_output_path)
                    existing_df.to_csv(score_backup_path, index=False)
                    logger.info(f"Created backup of existing score results: {score_backup_path}")
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")
            
            # Save new aggregated results
            all_que_results.to_csv(score_output_path, index=False)
            logger.info(f"Saved aggregated score results to: {score_output_path}")
            logger.info(f"Total score records: {len(all_que_results)}")
        else:
            logger.warning("No score results to aggregate")
        
        # Generate summary report
        summary_report = generate_aggregation_summary(
            all_companies, companies_with_prompts, companies_with_scores,
            total_prompt_records, total_score_records, all_prompt_results, all_que_results
        )
        
        # Save summary report
        summary_path = os.path.join(output_dir, f'aggregation_summary_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        logger.info(f"Saved aggregation summary to: {summary_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("AGGREGATION SUMMARY")
        print("="*60)
        print(summary_report)
        print("="*60)
        
        logger.info(f"Successfully aggregated results from {len(all_companies)} companies")
        return all_prompt_results, all_que_results
        
    except Exception as e:
        logger.error(f"Error in aggregation process: {e}")
        import traceback
        logger.error(f"Aggregation error traceback: {traceback.format_exc()}")
        return None, None

def generate_aggregation_summary(all_companies, companies_with_prompts, companies_with_scores, 
                               total_prompt_records, total_score_records, all_prompt_results, all_que_results):
    """Generate a summary report of the aggregation process"""
    
    summary = []
    summary.append(f"Aggregation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    summary.append("OVERVIEW:")
    summary.append(f"  - Total companies found: {len(all_companies)}")
    summary.append(f"  - Companies with prompt results: {companies_with_prompts}")
    summary.append(f"  - Companies with score results: {companies_with_scores}")
    summary.append(f"  - Total prompt records: {total_prompt_records}")
    summary.append(f"  - Total score records: {total_score_records}")
    summary.append("")
    
    summary.append("COMPANIES PROCESSED:")
    for company in sorted(all_companies):
        summary.append(f"  - {company}")
    summary.append("")
    
    if not all_prompt_results.empty:
        summary.append("PROMPT RESULTS ANALYSIS:")
        # Analyze by category
        if 'cat' in all_prompt_results.columns:
            cat_counts = all_prompt_results['cat'].value_counts()
            summary.append("  Results by category:")
            for cat, count in cat_counts.items():
                summary.append(f"    - Category {cat}: {count} records")
        
        # Analyze by company
        if 'company' in all_prompt_results.columns:
            company_counts = all_prompt_results['company'].value_counts()
            summary.append("  Results by company:")
            for company, count in company_counts.items():
                summary.append(f"    - {company}: {count} records")
        summary.append("")
    
    if not all_que_results.empty:
        summary.append("SCORE RESULTS ANALYSIS:")
        # Analyze score distribution
        if 'score' in all_que_results.columns:
            score_stats = all_que_results['score'].describe()
            summary.append("  Score statistics:")
            summary.append(f"    - Mean score: {score_stats['mean']:.2f}")
            summary.append(f"    - Median score: {score_stats['50%']:.2f}")
            summary.append(f"    - Min score: {score_stats['min']:.2f}")
            summary.append(f"    - Max score: {score_stats['max']:.2f}")
        
        # Analyze by category
        if 'category' in all_que_results.columns:
            cat_counts = all_que_results['category'].value_counts()
            summary.append("  Scores by category:")
            for cat, count in cat_counts.items():
                summary.append(f"    - Category {cat}: {count} scores")
        
        # Analyze by company
        if 'company' in all_que_results.columns:
            company_counts = all_que_results['company'].value_counts()
            summary.append("  Scores by company:")
            for company, count in company_counts.items():
                summary.append(f"    - {company}: {count} scores")
    
    return "\n".join(summary)

# Update the main function to include the aggregate option
def main():
    """Main function to demonstrate the usage of the Corporate Governance Agent"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Corporate Governance Scoring System')
    parser.add_argument('--company', type=str, help='Company symbol (e.g., PAYTM)')
    parser.add_argument('--path', type=str, help='Base path for company data')
    parser.add_argument('--mode', type=str, choices=['setup', 'process', 'score', 'all', 'aggregate'], default='all',
                      help='Operation mode (setup, process, score, all, or aggregate)')
    parser.add_argument('--category', type=int, choices=[1, 2, 3, 4], help='Category to score (1-4)')
    parser.add_argument('--question', type=int, help='Specific question number to process/score')
    parser.add_argument('--fresh', action='store_true', help='Process all questions from scratch')
    parser.add_argument('--retrieval', type=str, choices=['hybrid', 'bm25', 'vector', 'direct'], 
                      default='hybrid', help='Retrieval method to use')
    parser.add_argument('--parent-path', type=str, help='Parent path containing all company directories (for aggregation)')
    
    args = parser.parse_args()
    
    print(args)
    
    try:
        # Handle aggregation mode separately
        if args.mode == 'aggregate':
            if args.parent_path:
                parent_path = args.parent_path
            elif args.path:
                # If path is provided, use its parent directory
                parent_path = os.path.dirname(args.path)
            elif args.company:
                # Try to infer parent path from company path structure
                default_company_path = f'/Users/monilshah/Documents/GitHub/AgentEval/{args.company}/'
                parent_path = os.path.dirname(default_company_path)
            else:
                # Use default parent path
                parent_path = '/Users/monilshah/Documents/GitHub/AgentEval/'
            
            logger.info(f"Running aggregation mode with parent path: {parent_path}")
            all_prompt_results, all_que_results = aggregate_all_companies(parent_path)
            
            if all_prompt_results is not None or all_que_results is not None:
                logger.info("Aggregation completed successfully")
                return 0
            else:
                logger.error("Aggregation failed")
                return 1
        
        # For all other modes, company is required
        if not args.company:
            logger.error("Company symbol is required for non-aggregation modes")
            return 1
        
        # Create configuration and set retrieval method
        config = CGSConfig(args.company)
        if args.path:
            config.base_path = args.path
        
        # Set the retrieval method from command line argument
        config.retrieval_method = args.retrieval
        
        agent = CorporateGovernanceAgent(args.company, base_path=args.path, config=config)
        logger.info(f"Agent initialized for company {args.company} with retrieval method: {args.retrieval}")
        
        # Determine operation mode
        if args.mode == 'setup' or args.mode == 'all':
            agent.setup()
            logger.info("Setup completed")
        
        if args.mode == 'process' or args.mode == 'all':
            if args.question:
                agent.process_questions(load_all_fresh=args.fresh, sr_no_list=[args.question])
                logger.info(f"Processed question {args.question}")
            else:
                agent.process_questions(load_all_fresh=args.fresh)
                logger.info("Processed all questions")
        
        if args.mode == 'score' or args.mode == 'all':
            if args.question:
                agent.score_topic(args.question)
                logger.info(f"Scored topic {args.question}")
            elif args.category:
                agent.score_category(args.category)
                logger.info(f"Scored category {args.category}")
            else:
                agent.score_all_categories()
                logger.info("Scored all categories")
        
        logger.info("All operations completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


def test_retrieval(company_sym, document_path, query):
    """Test just the retrieval functionality"""
    # Initialize configuration
    config = CGSConfig(company_sym)
    config.retrieval_method = "bm25"
    
    # Initialize components
    document_processor = DocumentProcessor(config)
    
    # Create page-level chunks
    chunks = document_processor.create_page_level_chunks(document_path)
    print(f"Created {len(chunks)} chunks")
    
    # Create BM25 retriever
    from langchain_community.retrievers import BM25Retriever
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 5  # Return top 5 results
    
    # Perform search
    results = retriever.get_relevant_documents(query)
    
    # Print results
    print(f"\nTop {len(results)} results for query: {query}\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Text snippet: {doc.page_content[:200]}...\n")
    
    return results


def test_gemini_api(company_sym, query, text_content):
    """Test Gemini API call with text only"""
    # Initialize configuration
    config = CGSConfig(company_sym)
    
    try:
        # Initialize Gemini client
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            print("No Google API key found in environment")
            return
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content with text input
        print(f"Sending query to Gemini: {query}")
        print(f"With text content (first 100 chars): {text_content[:100]}...")
        
        response = model.generate_content([
            "Please answer the following question based on the provided text excerpts:",
            query,
            "Text excerpts:",
            text_content
        ])
        
        print("\n--- Gemini Response ---\n")
        print(response.text)
        
        return response.text
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# Add to main check
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-gemini":
            company = "PAYTM"
            query = "What is the shareholding pattern and who are the top 10 shareholders?"
            text_content = "Sample content about shareholding pattern..."  # Replace with real content
            test_gemini_api(company, query, text_content)
            sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--test-retrieval":
        company = "PAYTM"
        document_path = f"/Users/monilshah/Documents/02_NWU/01_capstone/04_Code_v3/{company}/98_data/annual_report_url.pdf"
        query = "shareholding pattern top 10 shareholders"
        test_retrieval(company, document_path, query)
        sys.exit(0)
    sys.exit(main())
    
    

## Sample commands to run the script

# # Set up the documents for a company
# python scoring_topics_agentic_langchain.py --company PAYTM --mode setup

# # Process all questions for a company
# python scoring_topics_agentic_langchain.py --company PAYTM --mode process

# # Process a specific question
# python scoring_topics_agentic_langchain.py --company PAYTM --mode process --question 43

# # Score a specific topic
# python scoring_topics_agentic_langchain.py --company PAYTM --mode score --question 10

# # Score an entire category
# python scoring_topics_agentic_langchain.py --company PAYTM --mode score --category 1

# # Do everything in one command
# python scoring_topics_agentic_langchain.py --company PAYTM --mode all    

# # Use BM25-only retrieval
# python scoring_topics_agentic_langchain.py --company PAYTM --mode process --question 43 --retrieval bm25

# # Use hybrid retrieval (default)
# python scoring_topics_agentic_langchain.py --company PAYTM --mode process --question 43

# # Use vector-only retrieval
# python scoring_topics_agentic_langchain.py --company PAYTM --mode process --question 43 --retrieval vector


# # Aggregate all companies in the default parent directory
# python scoring_topics_agentic_langchain.py --mode aggregate

# # Aggregate with custom parent path
# python scoring_topics_agentic_langchain.py --mode aggregate --parent-path /path/to/companies/

# # Aggregate using a specific company's path to infer parent
# python scoring_topics_agentic_langchain.py --mode aggregate --company PAYTM

