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
            self.base_path = f'/Users/monilshah/Documents/02_NWU/01_capstone/06_Code/{company_sym}/'
        
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
        self.model_to_use = 'llama3'
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
                embeddings = OllamaEmbeddings(model="llama3", embed_dim=384)  # Smaller embedding dimension
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama embeddings: {e}. Falling back to HuggingFace.")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Lighter model
            
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
    
    def setup_llm_clients(self):
        """Set up LLM clients for querying"""
        try:
            # Initialize Ollama client
            self.ollama = OllamaLLM(model=self.config.model_to_use)
            logger.info(f"Initialized Ollama with model {self.config.model_to_use}")
            
            # Initialize Gemini client
            self.gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            logger.info("Initialized Gemini client")
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {e}")
    
    def query_document(self, document_path: str, prompt: str, use_rag: bool = True):
        """Query a document with a given prompt"""
        logger.info(f"Querying document {document_path} with prompt: {prompt}")
        
        try:
            # Check if we're using "direct" retrieval method
            if self.config.retrieval_method.lower() == "direct":
                # Skip RAG and go directly to Gemini query
                logger.info(f"Using direct method as configured, bypassing RAG")
                result = self._query_with_gemini(document_path, prompt)
                if result and len(result.strip()) > 0:
                    return result
                    
                # If that fails, try ChatPDF as final fallback
                if self.config.CHATPDF_API_KEY:
                    result = self._query_with_chatpdf(document_path, prompt)
                    return result
                    
                return "Unable to process document query with direct method."
                
            # For non-direct methods, proceed with RAG if enabled
            if use_rag:
                result = self._query_with_rag(document_path, prompt)
                if result and len(result.strip()) > 0:
                    return result
            
            # Fallback to direct Gemini query
            result = self._query_with_gemini(document_path, prompt)
            if result and len(result.strip()) > 0:
                return result
            
            # Final fallback to ChatPDF if configured
            if self.config.CHATPDF_API_KEY:
                result = self._query_with_chatpdf(document_path, prompt)
                return result
                
            return "Unable to process document query."
        except Exception as e:
            logger.error(f"Error querying document: {e}")
            return f"Error querying document: {str(e)}"
        
    def _query_with_rag(self, document_path, prompt):
        """Query using optimized hybrid RAG with PDF slices"""
        try:
            # Check if we're using direct method
            if self.config.retrieval_method.lower() == "direct":
                logger.info(f"Direct method specified. Bypassing RAG for {document_path}")
                return None  # Signal caller to use direct method instead
                
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
                return None
            
            # Get relevant documents using the modern API
            try:
                # First try the new invoke method
                top_chunks = retriever.invoke(prompt)
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
                        return None
            
            if not top_chunks:
                logger.warning(f"No relevant chunks found for query in {document_path}")
                
                # Fallback to using first few chunks if we have them cached
                if document_path in chunks_cache and chunks_cache[document_path]:
                    logger.info("Using first few chunks as fallback since no relevant chunks found")
                    top_chunks = chunks_cache[document_path][:5]  # Use first 5 chunks as fallback
                else:
                    return None
            
            # Limit to at most 10 chunks to keep processing time reasonable
            if len(top_chunks) > 10:
                top_chunks = top_chunks[:10]
            
            # Query with PDF slices
            result = self.query_with_pdf_slices(prompt, top_chunks)
            
            logger.info(f"RAG query completed for {document_path} with {len(top_chunks)} chunks")
            return result
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
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
    """Handles verification of LLM outputs"""
    
    def __init__(self, config: CGSConfig):
        self.config = config
        
        # Initialize LLM
        try:
            from langchain_ollama import OllamaLLM 
            self.ollama = OllamaLLM(model=config.model_to_use)
            logger.info(f"Initialized Ollama with model {config.model_to_use}")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            self.ollama = None  # Ensure attribute exists even if initialization fails
    
    def verify_answer_quality(self, query: str, answer: str) -> Dict[str, str]:
        """Verify if the answer is good quality and has proper citations with robust parsing"""
        
        if not hasattr(self, 'ollama') or self.ollama is None:
            logger.warning("Ollama LLM not available, using simplified verification")
            # Simple fallback checks
            got_answer = "no" if "could not process document" in answer.lower() or "error" in answer.lower() else "yes"
            source_mentioned = "yes" if "page" in answer.lower() or "source" in answer.lower() else "no"
            return {"got_answer": got_answer, "source_mentioned": source_mentioned}
        
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
            # Try the chain approach
            raw_result = self.ollama.invoke(prompt.format(
                query=query,
                answer=answer,
                format_instructions=custom_format_instructions
            ))
            
            logger.info("Received guardrail assessment, attempting to parse")
            
            # Try multiple parsing approaches
            try:
                # Attempt 1: Direct JSON parsing
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
                            
                logger.info(f"Successfully parsed guardrail result: {result_dict}")
                return result_dict
                    
            except Exception as parsing_error:
                logger.warning(f"Initial parsing failed: {parsing_error}, trying regex fallback")
                
                # Attempt 2: Use regex to extract JSON
                import re
                json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                matches = re.findall(json_pattern, raw_result, re.DOTALL)
                
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
                            
                # Attempt 3: Basic text analysis if JSON extraction failed
                got_answer = "yes" if any(x in raw_result.lower() for x in [
                    "provided a substantive answer",
                    "addresses the query",
                    "answered the question",
                    "got_answer\": \"yes",
                    "got_answer\":\"yes"
                ]) else "no"
                
                source_mentioned = "yes" if any(x in raw_result.lower() for x in [
                    "includes references",
                    "cites specific sources",
                    "page numbers",
                    "mentions sources",
                    "source_mentioned\": \"yes",
                    "source_mentioned\":\"yes"
                ]) else "no"
                
                logger.info(f"Text analysis extraction: got_answer={got_answer}, source_mentioned={source_mentioned}")
                return {"got_answer": got_answer, "source_mentioned": source_mentioned}
                
        except Exception as e:
            logger.error(f"Error in guardrail verification: {e}")
                
        # Simple fallback if all else fails
        got_answer = "no" if "could not process document" in answer.lower() or "error" in answer.lower() else "yes"
        source_mentioned = "yes" if "page" in answer.lower() or "source" in answer.lower() else "no"
        
        logger.info(f"Using simple heuristic fallback: got_answer={got_answer}, source_mentioned={source_mentioned}")
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
        else:
            return original_query
            
        try:
            modified_query = self.ollama.invoke(prompt)
            logger.info(f"Modified query: {modified_query}")
            return modified_query.strip()
        except Exception as e:
            logger.error(f"Error modifying query: {e}")
            return original_query

class ScoringAgent:
    """Handles scoring of answers based on defined criteria"""
    
    def __init__(self, config: CGSConfig):
        self.config = config
        
        # Initialize LLM
        try:
            self.ollama = OllamaLLM(model=config.model_to_use)
            logger.info(f"Initialized Ollama with model {config.model_to_use}")
            
            self.gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            logger.info("Initialized Gemini client")
        except Exception as e:
            logger.error(f"Error initializing LLMs: {e}")
    
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
    
    def score_answer_unused(self, scoring_criteria: str, content: str) -> Dict[str, Union[int, str]]:
        """Score the content based on provided criteria"""
        
        # Define the output schema
        class ScoreOutput(BaseModel):
            score: int = Field(description="Score based on the criteria (integer)")
            justification: str = Field(description="Detailed justification for the score")
            
        parser = JsonOutputParser(pydantic_object=ScoreOutput)
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_template(
            """
            You are a corporate governance scoring expert.
            
            You need to score the following content based on specific criteria.
            
            Scoring Criteria:
            {scoring_criteria}
            
            Content to Score:
            {content}
            
            Important guidelines:
            - Do not assume any details not present in the content
            - If relevant information is missing, score should be 0
            - Include specific references to page numbers and document names in your justification
            - Be thorough in explaining why the content meets or fails to meet the criteria
            
            {format_instructions}
            """
        )
        
        # Create the chain
        chain = (
            {"scoring_criteria": lambda x: x[0], 
             "content": lambda x: x[1],
             "format_instructions": lambda _: parser.get_format_instructions()}
            | prompt
            | self.ollama
            | parser
        )
        
        try:
            # Run the chain
            result = chain.invoke([scoring_criteria, content])
            logger.info(f"Scoring results: score={result.score}")
            return {"score": result.score, "justification": result.justification}
        except Exception as e:
            logger.error(f"Error in scoring: {e}")
            
            # Fallback to regex parsing if JSON parsing fails
            try:
                raw_result = self.ollama.invoke(prompt.format(
                    scoring_criteria=scoring_criteria,
                    content=content,
                    format_instructions=parser.get_format_instructions()
                ))
                
                # Try to extract JSON with regex
                match = re.search(r'\{.*\}', raw_result, re.DOTALL)
                if match:
                    response = json.loads(match.group())
                    return {
                        "score": response.get('score', 0),
                        "justification": response.get('justification', 'Could not determine justification')
                    }
            except:
                pass
                
            # Default fallback
            return {"score": 0, "justification": f"Error in scoring: {str(e)}"}

    def score_answer(self, scoring_criteria: str, content: str) -> Dict[str, Union[int, str]]:
        """Score the content based on provided criteria with robust error handling"""
        
        # Define the output schema
        class ScoreOutput(BaseModel):
            score: int = Field(description="Score based on the criteria (integer from 0-10)")
            justification: str = Field(description="Detailed justification for the score")
        
        # Create a custom explicit formatting instruction
        custom_format_instructions = """
        You must respond with a valid JSON object using exactly this format:
        {
            "score": <integer_between_0_and_10>,
            "justification": "<detailed_explanation_with_evidence>"
        }
        
        The score must be an integer number between 0 and 10.
        Do not include any other text, explanation, or formatting outside of this JSON object.
        """
        
        # Initialize the parser
        parser = JsonOutputParser(pydantic_object=ScoreOutput)
        
        # Create the prompt with explicit formatting instructions
        prompt = ChatPromptTemplate.from_template(
            """
            You are a corporate governance scoring expert.
            
            You need to score the following content based on specific criteria.
            
            Scoring Criteria:
            {scoring_criteria}
            
            Content to Score:
            {content}
            
            Important guidelines:
            - Do not assume any details not present in the content
            - If relevant information is missing, score should be 0
            - Include specific references to page numbers and document names in your justification
            - Be thorough in explaining why the content meets or fails to meet the criteria
            - Score must be between 0 and 10, with 10 being the highest
            
            {format_instructions}
            
            Remember: Your response must be ONLY a valid JSON object with the keys "score" (integer) and "justification" (string).
            """
        )
        
        # Create the chain with format instructions
        chain = (
            {"scoring_criteria": lambda x: x[0], 
            "content": lambda x: x[1],
            "format_instructions": lambda _: custom_format_instructions}
            | prompt
            | self.ollama
        )
        
        try:
            # First attempt: Run the chain and try structured parsing
            raw_result = chain.invoke([scoring_criteria, content])
            logger.info(f"Raw scoring result received, attempting to parse...")
            print(raw_result)
            
            # Try multiple parsing approaches
            try:
                # Attempt 1: Direct JSON parsing of the entire response
                if isinstance(raw_result, str):
                    # Clean the string to extract just the JSON part
                    cleaned_result = raw_result.strip()
                    # If the response has text before/after the JSON, try to extract just the JSON part
                    json_start = cleaned_result.find('{')
                    json_end = cleaned_result.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = cleaned_result[json_start:json_end]
                        result_dict = json.loads(json_str)
                    else:
                        # If no JSON delimiters found, try parsing the whole string
                        result_dict = json.loads(cleaned_result)
                elif isinstance(raw_result, dict):
                    # Already a dictionary, use directly
                    result_dict = raw_result
                else:
                    # Try to access as object attributes
                    result_dict = {
                        "score": getattr(raw_result, "score", 0),
                        "justification": getattr(raw_result, "justification", "No justification provided")
                    }
                    
                # Ensure we have both required fields with correct types
                if "score" not in result_dict:
                    result_dict["score"] = 0
                if "justification" not in result_dict:
                    result_dict["justification"] = "No justification provided"
                    
                # Ensure score is an integer
                try:
                    result_dict["score"] = int(result_dict["score"])
                except (ValueError, TypeError):
                    logger.warning(f"Score was not an integer: {result_dict.get('score')}, defaulting to 0")
                    result_dict["score"] = 0
                    
                # Ensure score is in valid range (0-10)
                if result_dict["score"] < 0 or result_dict["score"] > 10:
                    logger.warning(f"Score out of range: {result_dict['score']}, clamping to 0-10")
                    result_dict["score"] = max(0, min(10, result_dict["score"]))
                    
                logger.info(f"Successfully parsed scoring result: score={result_dict['score']}")
                return result_dict
                
            except Exception as parsing_error:
                logger.warning(f"Initial parsing failed: {parsing_error}, trying regex fallback")
                
                # Attempt 2: Use regex to extract JSON
                import re
                json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                matches = re.findall(json_pattern, raw_result, re.DOTALL)
                
                if matches:
                    for potential_json in matches:
                        try:
                            result_dict = json.loads(potential_json)
                            if "score" in result_dict and "justification" in result_dict:
                                # Ensure score is an integer
                                try:
                                    result_dict["score"] = int(result_dict["score"])
                                except (ValueError, TypeError):
                                    result_dict["score"] = 0
                                    
                                # Ensure score is in valid range (0-10)
                                result_dict["score"] = max(0, min(10, result_dict["score"]))
                                
                                logger.info(f"Regex extraction successful: score={result_dict['score']}")
                                return result_dict
                        except json.JSONDecodeError:
                            continue
                
                # Attempt 3: Basic regex to find score if JSON extraction failed
                score_pattern = r'score[^\d]*(\d+)'
                score_match = re.search(score_pattern, raw_result, re.IGNORECASE)
                if score_match:
                    score = int(score_match.group(1))
                    # Clamp to valid range
                    score = max(0, min(10, score))
                    logger.info(f"Basic score extraction: {score}")
                    return {
                        "score": score,
                        "justification": f"Extracted from partial parsing. Original output: {raw_result}..."
                    }
        except Exception as e:
            logger.error(f"Error in main scoring process: {e}")
        
        # If all else fails, try a simpler direct approach as final fallback
        try:
            # Create a much simpler prompt focused just on getting a valid result
            simple_prompt = f"""
            Rate the following content on a scale of 0-10 based on these criteria:
            {scoring_criteria}
            
            Content:
            {content}
            
            Respond ONLY with a valid JSON object containing 'score' (integer 0-10) and 'justification' (string).
            Example: {{"score": 5, "justification": "Explanation here"}}
            """
            
            simple_result = self.ollama.invoke(simple_prompt)
            
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', simple_result, re.DOTALL)
            if json_match:
                try:
                    result_dict = json.loads(json_match.group())
                    if "score" in result_dict:
                        # Ensure score is an integer in valid range
                        try:
                            score = int(result_dict["score"])
                            score = max(0, min(10, score))
                        except (ValueError, TypeError):
                            score = 0
                            
                        return {
                            "score": score,
                            "justification": result_dict.get("justification", "No detailed justification provided.")
                        }
                except json.JSONDecodeError:
                    pass
        except Exception as fallback_error:
            logger.error(f"Final fallback failed: {fallback_error}")
        
        # Ultimate fallback - if everything else fails
        logger.error(f"All parsing methods failed. Returning default score 0.")
        return {
            "score": 0,
            "justification": "Failed to parse the scoring result. The scoring system encountered technical issues."
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
        """Process questions from prompts.csv and get answers"""
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
                
                # Query the document
                result = self.query_engine.query_document(source_path, message)
                
                if not result:
                    logger.warning(f"No result obtained for {source_path}")
                    result = "Could not process document."
                
                # Apply guardrails
                try:
                    guardrail_result = self.guardrail_agent.verify_answer_quality(message, result)
                    
                    # Convert to string values for consistency
                    if guardrail_result and isinstance(guardrail_result, dict):
                        got_answer = guardrail_result.get('got_answer', 'no')
                        source_mentioned = guardrail_result.get('source_mentioned', 'no')
                    else:
                        # Handle case where guardrail_result is an object with attributes
                        try:
                            got_answer = getattr(guardrail_result, 'got_answer', 'no')
                            source_mentioned = getattr(guardrail_result, 'source_mentioned', 'no')
                        except:
                            got_answer = 'no'
                            source_mentioned = 'no'
                except Exception as e:
                    logger.error(f"Error in guardrail verification: {e}")
                    # Fallback to simple checks if guardrail fails
                    got_answer = "no" if "could not process document" in result.lower() or "error" in result.lower() else "yes"
                    source_mentioned = "yes" if "page" in result.lower() or "source" in result.lower() else "no"
                
                # Try to improve answer if needed
                if got_answer == 'no':
                    try:
                        # Modify the query to try to get a better answer
                        modified_query = self.guardrail_agent.modify_query(message, result, "answer")
                        logger.info(f"Modified query for better answer: {modified_query}")
                        
                        # Query again with modified query
                        result_attempt2 = self.query_engine.query_document(source_path, modified_query)
                        if result_attempt2:
                            result = result + "\n\n" + result_attempt2
                    except Exception as e:
                        logger.error(f"Error in follow-up query for better answer: {e}")
                
                # Try to get source references if missing
                if source_mentioned == 'no':
                    try:
                        # Modify the query to try to get source references
                        modified_query = self.guardrail_agent.modify_query(message, result, "source")
                        logger.info(f"Modified query for source references: {modified_query}")
                        
                        # Query again with modified query
                        result_attempt2 = self.query_engine.query_document(source_path, modified_query)
                        if result_attempt2:
                            result = result + "\n\n" + result_attempt2
                    except Exception as e:
                        logger.error(f"Error in follow-up query for source references: {e}")
                
                # Save result
                run_time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_row = pd.DataFrame({
                    'run_time_stamp': [run_time_stamp],
                    'sr_no': [sr_no],
                    'cat': [cat],
                    'que_no': [que_no],
                    'source': [source_filter],
                    'message': [message],
                    'result': [result]
                })
                
                # Append to results
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                # Save incrementally
                try:
                    results_df.to_csv(results_path, index=False)
                    logger.info(f"Saved result for question {que_no}, source {os.path.basename(source_path)}")
                except Exception as e:
                    logger.error(f"Error saving results to CSV: {e}")
        
        return results_df
    
    def score_topic(self, topic_no):
        """Score a specific topic based on predefined criteria"""
        logger.info(f"Scoring topic {topic_no}")
        
        # Load scoring criteria
        criteria_path = os.path.join(self.config.parent_path, 'scoring_creteria.csv')
        if not os.path.exists(criteria_path):
            logger.error(f"Scoring criteria file not found: {criteria_path}")
            return
            
        sc_df = pd.read_csv(criteria_path)
        topic_criteria = sc_df[sc_df['topic_no'] == topic_no]
        
        if topic_criteria.empty:
            logger.error(f"No scoring criteria found for topic {topic_no}")
            return
            
        scoring_criteria = topic_criteria['scoring_criteria'].iloc[0]
        
        # Get content for the topic
        content, cat = self._get_topic_content(topic_no)
        
        if not content or not content.strip():
            logger.warning(f"No content available for topic {topic_no}")
            return
        
        # Apply any special post-processing
        content = self.scoring_agent.postprocess_content(topic_no, content)
        
        # Score the content
        score_result = self.scoring_agent.score_answer(scoring_criteria, content)
        
        # Save the score
        self._save_score(topic_no, cat, score_result['score'], score_result['justification'])
        
        return score_result
    
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

def main():
    """Main function to demonstrate the usage of the Corporate Governance Agent"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Corporate Governance Scoring System')
    parser.add_argument('--company', type=str, required=True, help='Company symbol (e.g., PAYTM)')
    parser.add_argument('--path', type=str, help='Base path for company data')
    parser.add_argument('--mode', type=str, choices=['setup', 'process', 'score', 'all'], default='all',
                      help='Operation mode (setup, process, score, or all)')
    parser.add_argument('--category', type=int, choices=[1, 2, 3, 4], help='Category to score (1-4)')
    parser.add_argument('--question', type=int, help='Specific question number to process/score')
    parser.add_argument('--fresh', action='store_true', help='Process all questions from scratch')
    
    parser.add_argument('--retrieval', type=str, choices=['hybrid', 'bm25', 'vector', 'direct'], 
                      default='hybrid', help='Retrieval method to use')
    
    args = parser.parse_args()
    
    print(args)
    # Initialize the agent
    try:
        
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
                
                # Also aggregate results
                agent.aggregate_results()
                logger.info("Aggregated results")
        
        logger.info("All operations completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

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