from typing import List, Dict, Any, Optional, Union, Callable
import os
from llama_index_compat import (
    Document,
    SimpleDirectoryReader,
    service_context_compat,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    SimpleNodeParser,
    PDFReader,
    DocxReader,
    CSVReader,
    SimpleWebPageReader
)
from dyno_agent import DynoAgent

class DynoDataLoader:
    """Data loader utility for DynoAgent to interact with LlamaIndex."""
    
    def __init__(self, 
                persist_dir: Optional[str] = "./storage",
                embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the data loader with configuration.
        
        Args:
            persist_dir: Directory to persist index data
            embedding_model: Embedding model to use for vector storage
        """
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.service_context = service_context_compat.from_defaults()
        self.loaded_data = []
        self.index = None
        self.data_sources = {}
        self.custom_loaders = {}
        
        # Create persistence directory if it doesn't exist
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
    
    def register_custom_loader(self, source_type: str, loader_func: Callable) -> None:
        """Register a custom loader function for a specific source type."""
        self.custom_loaders[source_type] = loader_func
    
    def load_data(self, source_type: str, source_data: Any) -> List[Document]:
        """Load data from different source types."""
        if source_type in self.custom_loaders:
            # Use custom loader if registered
            documents = self.custom_loaders[source_type](source_data)
            self.loaded_data.extend(documents)
            self.data_sources[source_type] = source_data
            return documents
        
        if source_type == "directory":
            documents = self._load_from_directory(source_data)
        elif source_type == "pdf":
            documents = self._load_from_pdf(source_data)
        elif source_type == "docx":
            documents = self._load_from_docx(source_data)
        elif source_type == "csv":
            documents = self._load_from_csv(source_data)
        elif source_type == "webpage":
            documents = self._load_from_webpage(source_data)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        self.loaded_data.extend(documents)
        self.data_sources[source_type] = source_data
        return documents
    
    def _load_from_directory(self, directory_paths: List[str]) -> List[Document]:
        """Load documents from directories."""
        documents = []
        for directory in directory_paths:
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist.")
                continue
            
            reader = SimpleDirectoryReader(directory)
            docs = reader.load_data()
            documents.extend(docs)
        
        return documents
    
    def _load_from_pdf(self, pdf_paths: List[str]) -> List[Document]:
        """Load documents from PDF files."""
        reader = PDFReader()
        documents = []
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file {pdf_path} does not exist.")
                continue
            
            docs = reader.load_data(pdf_path)
            documents.extend(docs)
        
        return documents
    
    def _load_from_docx(self, docx_paths: List[str]) -> List[Document]:
        """Load documents from DOCX files."""
        reader = DocxReader()
        documents = []
        
        for docx_path in docx_paths:
            if not os.path.exists(docx_path):
                print(f"Warning: DOCX file {docx_path} does not exist.")
                continue
            
            docs = reader.load_data(docx_path)
            documents.extend(docs)
        
        return documents
    
    def _load_from_csv(self, csv_paths: List[str]) -> List[Document]:
        """Load documents from CSV files."""
        reader = CSVReader()
        documents = []
        
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file {csv_path} does not exist.")
                continue
            
            docs = reader.load_data(csv_path)
            documents.extend(docs)
        
        return documents
    
    def _load_from_webpage(self, urls: List[str]) -> List[Document]:
        """Load documents from web pages."""
        reader = SimpleWebPageReader()
        documents = reader.load_data(urls)
        return documents
    
    def create_index(self) -> VectorStoreIndex:
        """Create a vector store index from loaded documents."""
        if not self.loaded_data:
            raise ValueError("No documents loaded. Please load documents first.")
        
        # Handle both old and new versions of LlamaIndex
        try:
            # Try the new way (with settings)
            self.index = VectorStoreIndex.from_documents(
                documents=self.loaded_data
            )
        except TypeError:
            # Fall back to the old way (with service_context)
            self.index = VectorStoreIndex.from_documents(
                documents=self.loaded_data,
                service_context=self.service_context
            )
        return self.index
    
    def persist_index(self) -> None:
        """Persist the index to disk."""
        if self.index is None:
            raise ValueError("No index created. Please create an index first.")
        
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"Index persisted to {self.persist_dir}")
    
    def load_index(self) -> VectorStoreIndex:
        """Load an index from disk."""
        if not os.path.exists(self.persist_dir):
            raise ValueError(f"No index found at {self.persist_dir}")
        
        storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        self.index = load_index_from_storage(storage_context)
        return self.index
    
    def query_index(self, query_text: str) -> Dict[str, Any]:
        """Query the index with a text query."""
        if self.index is None:
            raise ValueError("No index available. Please create or load an index first.")
        
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        
        return {
            "answer": str(response),
            "source_nodes": [n.node.get_text() for n in response.source_nodes]
        } 