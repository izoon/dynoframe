from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import os
import json
import datetime
from ..core.dyno_agent_with_tools import DynoAgentWithTools
from ..dyno_llamaindex import DynoDataLoader
from ..llama_index_compat import Document, VectorStoreIndex
import openai
from ..llm_provider_factory import LLMProviderFactory
from ..llm_providers import LLMProvider
from ..task_complexity import TaskComplexityAnalyzer

class DynoRAGAgent(DynoAgentWithTools):
    """
    RAG (Retrieval-Augmented Generation) agent that retrieves relevant documents from a vector store 
    and uses them to augment the generation of responses.
    
    This agent combines the document handling capabilities of DynoAgentWithTools
    with LLM-powered generation for a complete RAG workflow.
    """
    
    def __init__(self, 
                 name: str, 
                 role: str, 
                 skills: List[str], 
                 goal: str,
                 llm_model: Optional[str] = None,
                 provider_name: str = "openrouter",
                 retrieval_k: int = 3,
                 similarity_threshold: float = 0.7,
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 enable_learning: bool = False,
                 learning_threshold: int = 10,
                 accuracy_boost_factor: float = 1.5,
                 use_rl_decision_agent: bool = True,
                 input_dependencies: Optional[List[Any]] = None,
                 tools_dataloaders: Optional[Dict[str, Callable]] = None,
                 index_name: str = "rag_index",
                 use_role_based_model: bool = True,
                 use_dynamic_parameters: bool = True,
                 config_path: str = "config.json"):
        """
        Initialize the RAG agent with configuration.
        
        Args:
            name: Agent name
            role: Agent role
            skills: List of skills
            goal: Agent goal
            llm_model: Default LLM model to use for generation (if None, will use role-based selection)
            provider_name: LLM provider name ("openrouter", "openai", "anthropic")
            retrieval_k: Number of documents to retrieve (default: 3)
            similarity_threshold: Threshold for similarity score filtering (default: 0.7)
            max_tokens: Maximum tokens for generation (default: 1000)
            temperature: Temperature for generation (default: 0.7)
            enable_learning: Whether to enable learning (default: False)
            learning_threshold: Learning threshold (default: 10)
            accuracy_boost_factor: Accuracy boost factor (default: 1.5)
            use_rl_decision_agent: Whether to use RL decision agent (default: True)
            input_dependencies: List of input dependencies (default: None)
            tools_dataloaders: Dictionary of tools dataloaders (default: None)
            index_name: Name of the index (default: rag_index)
            use_role_based_model: Whether to select model based on role (default: True)
            use_dynamic_parameters: Whether to adjust parameters based on task complexity (default: True)
            config_path: Path to configuration file (default: config.json)
        """
        # Call the parent constructor
        super().__init__(
            name=name,
            role=role,
            skills=skills,
            goal=goal,
            enable_learning=enable_learning,
            learning_threshold=learning_threshold,
            accuracy_boost_factor=accuracy_boost_factor,
            use_rl_decision_agent=use_rl_decision_agent,
            input_dependencies=input_dependencies,
            tools_dataloaders=tools_dataloaders,
            index_name=index_name
        )
        
        # Add RAG-specific skills
        self.skills.extend([
            "retrieval_augmented_generation",
            "context_aware_response",
            "document_based_reasoning"
        ])
        
        # RAG configuration
        self.provider_name = provider_name
        self.retrieval_k = retrieval_k
        self.similarity_threshold = similarity_threshold
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.use_role_based_model = use_role_based_model
        self.use_dynamic_parameters = use_dynamic_parameters
        
        # Initialize provider factory and get provider
        self.provider_factory = LLMProviderFactory(config_path=config_path)
        self.llm_provider = self.provider_factory.get_provider(provider_name)
        
        # Set model based on configuration
        if use_role_based_model:
            self.llm_model = self.llm_provider.get_model_for_role(role.lower())
        else:
            self.llm_model = llm_model or self.llm_provider.get_model_for_role("default")
            
        # Track RAG-specific metrics
        self.rag_history = []
        
        # Print provider info
        print(f"Using LLM provider: {self.llm_provider.get_provider_name()}")
        print(f"Selected model for role '{role}': {self.llm_model}")
        
        # Register RAG tools
        self._register_rag_tools()
    
    def _register_rag_tools(self) -> None:
        """Register RAG-specific tools."""
        rag_tools = {
            "retrieve_and_generate": self.retrieve_and_generate,
            "retrieve_documents": self.retrieve_documents,
            "generate_with_context": self.generate_with_context,
            "answer_question": self.answer_question,
            "build_context": self.build_context
        }
        
        for tool_name, tool_func in rag_tools.items():
            self.register_tool(tool_name, tool_func)
    
    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve (defaults to self.retrieval_k)
        
        Returns:
            List of retrieved documents
        """
        k = k or self.retrieval_k
        
        self.history.append({
            "task": "Retrieve documents",
            "context": query,
            "role": self.role
        })
        
        # Ensure index exists
        if self.index is None:
            if os.path.exists(self.index_storage_path):
                self.load_index()
            elif self.data_loader.loaded_data:
                self.create_vector_index()
            else:
                return []
        
        # Retrieve documents
        try:
            # Get query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=k
            )
            
            # Execute query
            response = query_engine.query(query)
            
            # Extract source nodes
            source_nodes = []
            if hasattr(response, "source_nodes"):
                source_nodes = response.source_nodes
            
            # Format results
            retrieved_docs = []
            for i, node in enumerate(source_nodes):
                doc = {
                    "id": i,
                    "text": node.node.get_text() if hasattr(node, "node") else node.get_text(),
                    "metadata": node.node.metadata if hasattr(node, "node") else node.metadata,
                    "similarity": node.score if hasattr(node, "score") else None
                }
                retrieved_docs.append(doc)
            
            # Filter by similarity threshold if scores are available
            if retrieved_docs and retrieved_docs[0].get("similarity") is not None:
                retrieved_docs = [doc for doc in retrieved_docs if doc.get("similarity", 0) >= self.similarity_threshold]
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    def build_context(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Build a context string from retrieved documents.
        
        Args:
            query: Original query
            documents: List of retrieved documents
        
        Returns:
            Context string
        """
        if not documents:
            return ""
        
        context = "Context information from retrieved documents:\n\n"
        
        for i, doc in enumerate(documents):
            doc_text = doc.get("text", "").strip()
            source = doc.get("metadata", {}).get("source", f"Document {i+1}")
            
            context += f"Document {i+1} (Source: {source}):\n{doc_text}\n\n"
        
        return context
    
    def generate_with_context(self, 
                             context: str, 
                             query: str, 
                             prompt_template: Optional[str] = None,
                             complexity: Optional[str] = None) -> str:
        """
        Generate a response using the LLM with context.
        
        Args:
            context: Context information
            query: User query
            prompt_template: Optional template for the prompt
            complexity: Optional task complexity override
            
        Returns:
            Generated response
        """
        # Default prompt template
        if prompt_template is None:
            prompt_template = """
            You are a helpful assistant named {agent_name} with the role of {agent_role}.
            Your goal is to {agent_goal}.
            
            Use the following context information to answer the user's question.
            If the information is not in the context, say that you don't know based on the available information.
            
            {context}
            
            User question: {query}
            
            {agent_name}'s response:
            """
        
        # Fill template
        prompt = prompt_template.format(
            agent_name=self.name,
            agent_role=self.role,
            agent_goal=self.goal,
            context=context,
            query=query
        )
        
        # Get generation parameters based on complexity and role
        if self.use_dynamic_parameters:
            if complexity is None:
                # Use TaskComplexityAnalyzer to determine complexity
                complexity = TaskComplexityAnalyzer.analyze_complexity(query, self.role)
                print(f"Detected task complexity: {complexity}")
            
            # Get parameters from the provider
            params = self.provider_factory.get_generation_params(complexity, self.role.lower())
        else:
            # Use default parameters
            params = {
                "model": self.llm_model,
                "max_tokens": self.default_max_tokens,
                "temperature": self.default_temperature
            }
        
        # Generate response using LLM provider
        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                **params
            )
            
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def retrieve_and_generate(self, 
                             query: str, 
                             k: Optional[int] = None, 
                             prompt_template: Optional[str] = None,
                             complexity: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve documents and generate a response in one step.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            prompt_template: Optional template for the prompt
            complexity: Optional task complexity override
        
        Returns:
            Dictionary with response and retrieved documents
        """
        self.history.append({
            "task": "Retrieve and generate",
            "context": query,
            "role": self.role
        })
        
        # Analyze task complexity if not provided and dynamic parameters are enabled
        if self.use_dynamic_parameters and complexity is None:
            complexity = TaskComplexityAnalyzer.analyze_complexity(query, self.role)
            print(f"Detected task complexity: {complexity}")
        
        # Retrieve documents
        documents = self.retrieve_documents(query, k)
        
        # Build context
        context = self.build_context(query, documents)
        
        # Generate response
        response = self.generate_with_context(context, query, prompt_template, complexity)
        
        # Track in RAG history
        rag_record = {
            "query": query,
            "documents": documents,
            "response": response,
            "timestamp": self._get_timestamp(),
            "complexity": complexity,
            "model_used": self.llm_model
        }
        self.rag_history.append(rag_record)
        
        return {
            "response": response,
            "documents": documents,
            "context": context,
            "complexity": complexity,
            "model_used": self.llm_model
        }
    
    def answer_question(self, 
                       question: str, 
                       k: Optional[int] = None, 
                       prompt_template: Optional[str] = None,
                       complexity: Optional[str] = None) -> str:
        """
        Answer a question using the RAG workflow (simplified interface).
        
        Args:
            question: User question
            k: Number of documents to retrieve
            prompt_template: Optional template for the prompt
            complexity: Optional task complexity override
        
        Returns:
            Generated answer
        """
        result = self.retrieve_and_generate(question, k, prompt_template, complexity)
        return result["response"]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.datetime.now().isoformat()
    
    def get_rag_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about RAG operations.
        
        Returns:
            Dictionary with statistics
        """
        if not self.rag_history:
            return {"error": "No RAG operations recorded yet"}
        
        total_queries = len(self.rag_history)
        avg_docs = sum(len(r["documents"]) for r in self.rag_history) / total_queries if total_queries > 0 else 0
        
        # Add model usage statistics
        model_usage = {}
        for record in self.rag_history:
            model = record.get("model_used", "unknown")
            model_usage[model] = model_usage.get(model, 0) + 1
        
        # Add complexity statistics if available
        complexity_stats = {}
        for record in self.rag_history:
            complexity = record.get("complexity")
            if complexity:
                complexity_stats[complexity] = complexity_stats.get(complexity, 0) + 1
        
        stats = {
            "total_rag_queries": total_queries,
            "average_documents_retrieved": avg_docs,
            "queries_per_session": total_queries / len(set(r.get("session_id", "default") for r in self.rag_history)) if total_queries > 0 else 0,
            "model_usage": model_usage
        }
        
        if complexity_stats:
            stats["complexity_distribution"] = complexity_stats
            
        return stats
    
    def save_rag_history(self, file_path: str) -> bool:
        """
        Save RAG history to a JSON file.
        
        Args:
            file_path: Path to save the history
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.rag_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving RAG history: {str(e)}")
            return False
    
    def load_rag_history(self, file_path: str) -> bool:
        """
        Load RAG history from a JSON file.
        
        Args:
            file_path: Path to load the history from
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                self.rag_history = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading RAG history: {str(e)}")
            return False 