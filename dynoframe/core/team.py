import asyncio
from typing import List, Dict, Set, Tuple, Any, Optional, Callable, Union
import networkx as nx
from .dyno_agent import DynoAgent
from ..pdf.pdf_processing_agent import PDFProcessingDecisionAgent
from .dyno_agent_with_tools import DynoAgentWithTools

class Team:
    """
    Team class that manages a group of agents and determines execution order based on dependencies.
    The team can execute agents in parallel or sequentially based on the dependency graph.
    """
    
    def __init__(self, name: str, agents: List[DynoAgent] = None, explicit_dependencies: Dict[str, List[str]] = None):
        """
        Initialize a team with a list of agents and optional explicit dependencies.
        
        Args:
            name: Name of the team
            agents: List of DynoAgent instances
            explicit_dependencies: Dictionary mapping agent names to lists of agent names they depend on
        """
        self.name = name
        self.agents = agents or []
        self.agent_map = {agent.name: agent for agent in self.agents}
        self.explicit_dependencies = explicit_dependencies or {}
        self.dependency_graph = nx.DiGraph()
        self.execution_plan = []
        self.results = {}
        
        # Add agents to the dependency graph
        for agent in self.agents:
            self.dependency_graph.add_node(agent.name, agent=agent)
        
        # Add explicit dependencies to the graph
        self._add_explicit_dependencies()
        
        # Analyze dependencies between agents
        self._analyze_dependencies()
        
        # Create the execution plan
        self._create_execution_plan()
    
    def add_agent(self, agent: DynoAgent, dependencies: List[str] = None) -> None:
        """
        Add an agent to the team with optional dependencies.
        
        Args:
            agent: The agent to add
            dependencies: List of agent names this agent depends on
        """
        self.agents.append(agent)
        self.agent_map[agent.name] = agent
        self.dependency_graph.add_node(agent.name, agent=agent)
        
        if dependencies:
            self.explicit_dependencies[agent.name] = dependencies
            for dep in dependencies:
                if dep in self.agent_map:
                    self.dependency_graph.add_edge(dep, agent.name)
        
        # Re-analyze dependencies and update execution plan
        self._analyze_dependencies()
        self._create_execution_plan()
    
    def _add_explicit_dependencies(self) -> None:
        """Add explicit dependencies to the dependency graph."""
        for agent_name, dependencies in self.explicit_dependencies.items():
            if agent_name in self.agent_map:
                for dep in dependencies:
                    if dep in self.agent_map:
                        self.dependency_graph.add_edge(dep, agent_name)
    
    def _analyze_dependencies(self) -> None:
        """
        Analyze dependencies between agents based on their input dependencies and tools.
        Adds edges to the dependency graph representing these dependencies.
        """
        # Check for input/output dependencies based on tools and dependencies
        for agent_a in self.agents:
            for agent_b in self.agents:
                if agent_a == agent_b:
                    continue
                
                # Check if agent_b needs any tool or data that agent_a provides
                self._check_tool_dependencies(agent_a, agent_b)
                
                # Check if agent_b has input dependencies that can be resolved by agent_a
                self._check_input_dependencies(agent_a, agent_b)
    
    def _check_tool_dependencies(self, provider: DynoAgent, consumer: DynoAgent) -> None:
        """
        Check if the consumer agent depends on tools provided by the provider agent.
        
        Args:
            provider: Agent that might provide tools
            consumer: Agent that might consume tools
        """
        # If both agents have tools_dataloaders attribute
        if hasattr(provider, 'tools_dataloaders') and hasattr(consumer, 'tools_dataloaders'):
            provider_tools = set(provider.tools_dataloaders.keys())
            
            # If consumer references provider's tools in its name or goal
            for tool in provider_tools:
                if (tool in consumer.name.lower() or 
                    (hasattr(consumer, 'goal') and tool in consumer.goal.lower())):
                    self.dependency_graph.add_edge(provider.name, consumer.name)
                    break
    
    def _check_input_dependencies(self, provider: DynoAgent, consumer: DynoAgent) -> None:
        """
        Check if the consumer agent has input dependencies that can be resolved by the provider agent.
        
        Args:
            provider: Agent that might provide input
            consumer: Agent that might consume input
        """
        # If consumer has input_dependencies attribute and provider's role or skills match
        if hasattr(consumer, 'input_dependencies') and consumer.input_dependencies:
            provider_role = provider.role.lower()
            provider_skills = [skill.lower() for skill in provider.skills]
            
            for dependency in consumer.input_dependencies:
                dep_type = type(dependency).__name__.lower()
                
                # If provider's role or skills indicate it can provide this dependency
                if (dep_type in provider_role or 
                    any(dep_type in skill for skill in provider_skills)):
                    self.dependency_graph.add_edge(provider.name, consumer.name)
                    break
    
    def _create_execution_plan(self) -> None:
        """
        Create an execution plan based on the dependency graph.
        Groups agents that can be executed in parallel.
        """
        try:
            # Check for cycles in the dependency graph
            if not nx.is_directed_acyclic_graph(self.dependency_graph):
                cycles = list(nx.simple_cycles(self.dependency_graph))
                raise ValueError(f"Dependency graph contains cycles: {cycles}")
            
            # Get topological sort (respects dependencies)
            topo_sort = list(nx.topological_sort(self.dependency_graph))
            
            # Group agents that can be executed in parallel
            self.execution_plan = []
            visited = set()
            
            for agent_name in topo_sort:
                if agent_name in visited:
                    continue
                
                # Find all agents at the same "level" (can be executed in parallel)
                level = set()
                for node in topo_sort:
                    if node in visited:
                        continue
                    
                    # If all node's dependencies are visited, it's at current level
                    if all(pred in visited for pred in self.dependency_graph.predecessors(node)):
                        level.add(node)
                
                self.execution_plan.append(list(level))
                visited.update(level)
        
        except Exception as e:
            print(f"Error creating execution plan: {str(e)}")
            # Fallback to sequential execution
            self.execution_plan = [[agent.name] for agent in self.agents]
    
    def execute_sequential(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute all agents sequentially based on the execution plan.
        
        Args:
            context: Initial context for the agents
        
        Returns:
            Dictionary of results from all agents
        """
        context = context or {}
        results = {}
        
        for level in self.execution_plan:
            for agent_name in level:
                agent = self.agent_map.get(agent_name)
                if agent:
                    print(f"Executing {agent_name} sequentially")
                    result = agent.perform_task(f"Execute {agent_name}", context)
                    results[agent_name] = result
                    context[agent_name] = result
        
        self.results = results
        return results
    
    async def execute_parallel(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute agents in parallel based on the execution plan.
        Agents within each level are executed in parallel, while levels are executed sequentially.
        
        Args:
            context: Initial context for the agents
        
        Returns:
            Dictionary of results from all agents
        """
        context = context or {}
        results = {}
        
        for level in self.execution_plan:
            level_tasks = []
            
            for agent_name in level:
                agent = self.agent_map.get(agent_name)
                if agent:
                    print(f"Preparing {agent_name} for parallel execution")
                    task = asyncio.create_task(self._execute_agent_async(agent, agent_name, context))
                    level_tasks.append((agent_name, task))
            
            # Wait for all tasks in this level to complete
            for agent_name, task in level_tasks:
                result = await task
                results[agent_name] = result
                context[agent_name] = result
        
        self.results = results
        return results
    
    async def _execute_agent_async(self, agent: DynoAgent, agent_name: str, context: Dict[str, Any]) -> Any:
        """
        Execute an agent asynchronously.
        
        Args:
            agent: The agent to execute
            agent_name: Name of the agent
            context: Context for the agent
        
        Returns:
            Result from the agent
        """
        print(f"Executing {agent_name} in parallel")
        return await asyncio.to_thread(agent.perform_task, f"Execute {agent_name}", context)
    
    def execute_optimal(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the team using the optimal execution strategy based on the dependency structure.
        
        If all agents are in a single level, executes them in parallel.
        If all levels have only one agent, executes them sequentially.
        Otherwise, uses a mix of parallel and sequential execution.
        
        Args:
            context: Initial context for the agents
        
        Returns:
            Dictionary of results from all agents
        """
        # If we only have one level in the execution plan, execute all in parallel
        if len(self.execution_plan) == 1:
            return self.execute_parallel(context)
        
        # If all levels have only one agent, execute sequentially
        if all(len(level) == 1 for level in self.execution_plan):
            return self.execute_sequential(context)
        
        # Otherwise, use the mixed approach
        return self.execute_parallel(context)
    
    def visualize_dependencies(self, output_file: str = "team_dependencies.png") -> None:
        """
        Visualize the dependency graph using matplotlib and networkx.
        
        Args:
            output_file: Path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.dependency_graph)
            nx.draw(self.dependency_graph, pos, with_labels=True, node_color='lightblue', 
                   node_size=1500, arrows=True, arrowsize=20, font_size=10)
            plt.title(f"Team {self.name} Dependency Graph")
            plt.savefig(output_file)
            plt.close()
            print(f"Dependency graph saved to {output_file}")
        except ImportError:
            print("Could not visualize dependency graph. Please install matplotlib: pip install matplotlib")
    
    def get_execution_plan_str(self) -> str:
        """
        Get a string representation of the execution plan.
        
        Returns:
            String representation of the execution plan
        """
        plan_str = f"Execution plan for Team {self.name}:\n"
        for i, level in enumerate(self.execution_plan):
            if len(level) > 1:
                plan_str += f"Level {i+1} (Parallel): {', '.join(level)}\n"
            else:
                plan_str += f"Level {i+1} (Sequential): {level[0]}\n"
        return plan_str


# Example usage
if __name__ == "__main__":
    import asyncio
    from .dyno_agent import DynoAgent
    from ..pdf.pdf_processing_agent import PDFProcessingDecisionAgent, ImagePreprocessor, OCRPreprocessor
    
    # Create agents
    ocr_agent = DynoAgent(
        name="OCRAgent",
        role="Extractor",
        skills=["OCR", "Text Extraction"],
        goal="Extract text from documents"
    )
    
    analysis_agent = DynoAgent(
        name="AnalysisAgent",
        role="Analyzer",
        skills=["Text Analysis", "Summarization"],
        goal="Analyze and summarize text"
    )
    
    indexing_agent = DynoAgent(
        name="IndexingAgent",
        role="Indexer",
        skills=["Database", "Indexing"],
        goal="Index documents in a database"
    )
    
    pdf_agent = PDFProcessingDecisionAgent(
        name="PDFAgent",
        skills=["PDF Processing", "Format Detection"],
        goal="Process PDF documents"
    )
    
    # Create a team with explicit dependencies
    team = Team(
        name="DocumentProcessingTeam",
        agents=[ocr_agent, analysis_agent, indexing_agent, pdf_agent],
        explicit_dependencies={
            "AnalysisAgent": ["OCRAgent"],
            "IndexingAgent": ["AnalysisAgent"]
        }
    )
    
    # Print execution plan
    print(team.get_execution_plan_str())
    
    # Visualize dependencies
    team.visualize_dependencies()
    
    # Execute the team
    context = {"document_path": "example.pdf"}
    
    # Sequential execution
    print("\nExecuting sequentially...")
    sequential_results = team.execute_sequential(context)
    
    # Parallel execution
    print("\nExecuting in parallel...")
    parallel_results = asyncio.run(team.execute_parallel(context))
    
    # Optimal execution
    print("\nExecuting optimally...")
    optimal_results = team.execute_optimal(context) 