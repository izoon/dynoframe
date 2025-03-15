from typing import List, Dict, Any, Optional, Union, Callable
import os
from .dyno_agent import DynoAgent
from ..dyno_llamaindex import DynoDataLoader
from ..llama_index_compat import (
    Document,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex
)

class DynoAgentWithTools(DynoAgent):
    """Extended DynoAgent with LlamaIndex data loading and querying tools."""
    
    def __init__(self, name, role, skills, goal, 
                 enable_learning=False, 
                 learning_threshold=10, 
                 accuracy_boost_factor=1.5, 
                 use_rl_decision_agent=True,
                 input_dependencies: Optional[List[Any]] = None,
                 tools_dataloaders: Optional[Dict[str, Callable]] = None,
                 index_name: str = "dyno_index"):
        """Initialize the agent with LlamaIndex tools."""
        super().__init__(name, role, skills, goal, 
                         enable_learning, 
                         learning_threshold, 
                         accuracy_boost_factor, 
                         use_rl_decision_agent,
                         input_dependencies,
                         tools_dataloaders)
        
        # Initialize the data loader
        self.data_loader = DynoDataLoader()
        
        # Add tool skills
        self.skills.extend([
            "data_loading", 
            "document_indexing", 
            "information_retrieval"
        ])
        
        # Track indexed data sources
        self.indexed_sources = []
        
        self.index = None
        self.index_name = index_name
        self.index_storage_path = f"./data/{index_name}"
        
        # Register data loading tools to the agent
        self._register_data_tools()
    
    def _register_data_tools(self) -> None:
        """Register data processing tools to the agent."""
        # Register llama-index data loading methods as tools
        data_tools = {
            "load_directory_data": self.load_data_wrapper("directory"),
            "load_pdf_data": self.load_data_wrapper("pdf"),
            "load_docx_data": self.load_data_wrapper("docx"),
            "load_csv_data": self.load_data_wrapper("csv"),
            "load_webpage_data": self.load_data_wrapper("webpage"),
            "create_index": self.create_vector_index,
            "save_index": self.save_index,
            "load_index": self.load_index,
            "retrieve_info": self.retrieve_information
        }
        
        # Add these tools to the agent's tools
        for tool_name, tool_func in data_tools.items():
            self.register_tool(tool_name, tool_func)
    
    def load_data_wrapper(self, source_type: str) -> Callable:
        """Create a wrapper function for loading data of a specific type."""
        def load_data_func(source_data: Any) -> List[Document]:
            return self.load_data(source_type, source_data)
        return load_data_func
    
    def load_data(self, source_type: str, source_data: Any) -> List[Document]:
        """Load data using the data loader."""
        self.history.append({
            "task": "Load data",
            "context": f"Source type: {source_type}",
            "role": self.role
        })
        
        # Apply input dependencies if available
        processed_source_data = self._apply_dependencies_to_data(source_type, source_data)
        
        return self.data_loader.load_data(source_type, processed_source_data)
    
    def _apply_dependencies_to_data(self, source_type: str, source_data: Any) -> Any:
        """Apply input dependencies to preprocess the data source."""
        # If no dependencies, return the original data
        if not self.input_dependencies:
            return source_data
        
        processed_data = source_data
        
        # Apply each dependency in sequence
        for dependency in self.input_dependencies:
            # Check if dependency has a process_data method
            if hasattr(dependency, 'process_data') and callable(getattr(dependency, 'process_data')):
                try:
                    # Dependency should return processed data
                    result = dependency.process_data(source_type, processed_data)
                    if result is not None:
                        processed_data = result
                except Exception as e:
                    print(f"Error applying dependency {type(dependency).__name__} to {source_type}: {str(e)}")
        
        return processed_data
    
    def register_custom_loader(self, source_type: str, loader_func: Callable) -> None:
        """Register a custom loader for a specific data source type."""
        self.data_loader.register_custom_loader(source_type, loader_func)
        self.history.append({
            "task": "Register custom loader",
            "context": f"Source type: {source_type}",
            "role": self.role
        })
    
    def create_vector_index(self) -> bool:
        """Create a vector index from loaded documents."""
        self.history.append({
            "task": "Create vector index",
            "context": f"Document count: {len(self.data_loader.loaded_data)}",
            "role": self.role
        })
        
        if not self.data_loader.loaded_data:
            print("No documents loaded. Cannot create index.")
            return False
        
        try:
            # Handle both old and new versions of LlamaIndex
            try:
                # Try the new way (with settings)
                self.index = VectorStoreIndex.from_documents(
                    documents=self.data_loader.loaded_data
                )
            except TypeError:
                # Fall back to the old way (with service_context)
                self.index = VectorStoreIndex.from_documents(
                    documents=self.data_loader.loaded_data,
                    service_context=self.data_loader.service_context
                )
            return True
        except Exception as e:
            print(f"Error creating vector index: {str(e)}")
            return False
    
    def save_index(self) -> bool:
        """Save the index to disk."""
        self.history.append({
            "task": "Save index",
            "context": self.index_storage_path,
            "role": self.role
        })
        
        if self.index is None:
            print("No index to save.")
            return False
        
        try:
            # Create the storage directory if it doesn't exist
            os.makedirs(self.index_storage_path, exist_ok=True)
            
            # Save the index
            self.index.storage_context.persist(persist_dir=self.index_storage_path)
            return True
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self) -> bool:
        """Load the index from disk."""
        self.history.append({
            "task": "Load index",
            "context": self.index_storage_path,
            "role": self.role
        })
        
        if not os.path.exists(self.index_storage_path):
            print(f"Index storage path {self.index_storage_path} does not exist.")
            return False
        
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.index_storage_path)
            self.index = load_index_from_storage(storage_context)
            return True
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False
    
    def retrieve_information(self, query: str) -> Dict[str, Any]:
        """Retrieve information from the index."""
        self.history.append({
            "task": "Retrieve information",
            "context": query,
            "role": self.role
        })
        
        if self.index is None:
            # Attempt to load index if it exists
            if os.path.exists(self.index_storage_path):
                self.load_index()
            
            # If still no index, try to create one
            if self.index is None and self.data_loader.loaded_data:
                self.create_vector_index()
            
            # If still no index, return error
            if self.index is None:
                return {
                    "answer": "No index available. Please load data and create an index first.",
                    "source_nodes": []
                }
        
        try:
            # Handle both old and new versions of as_query_engine
            query_engine = self.index.as_query_engine()
            response = query_engine.query(query)
            
            # Handle different response formats
            source_nodes = []
            if hasattr(response, "source_nodes"):
                source_nodes = response.source_nodes
            
            return {
                "answer": str(response),
                "source_nodes": source_nodes
            }
        except Exception as e:
            print(f"Error retrieving information: {str(e)}")
            return {
                "answer": f"Error retrieving information: {str(e)}",
                "source_nodes": []
            }
    
    def get_document_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries of all loaded documents."""
        summaries = []
        
        for idx, doc in enumerate(self.data_loader.loaded_data):
            summary = {
                "id": idx,
                "text_preview": doc.text[:100] + "..." if len(doc.text) > 100 else doc.text,
                "metadata": doc.metadata
            }
            summaries.append(summary) 