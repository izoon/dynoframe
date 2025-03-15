class DynoAgent:
    """Base agent with dynamic role assignment, adaptation, and learning orchestration."""

    def __init__(self, name, role, skills, goal, enable_learning=False, learning_threshold=10, 
                 accuracy_boost_factor=1.5, use_rl_decision_agent=True, input_dependencies=None, 
                 tools_dataloaders=None):
        self.name = name
        self.role = role
        self.skills = skills  # Dynamic skills matrix
        self.goal = goal
        self.history = []  # Track past interactions
        self.human_feedback_scores = []  # Track human feedback (1-10)
        self.input_quality_scores = []  # Track self-assessment of input quality
        self.custom_metrics = {}  # Store user-defined metrics
        self.learning_data = []  # Store task execution history for learning
        self.learning_threshold = learning_threshold  # Number of inputs needed before adjusting sequencing
        self.enable_learning = enable_learning  # Learning activation switch
        self.accuracy_boost_factor = accuracy_boost_factor  # Factor to suggest higher input data for accuracy improvement
        self.execution_mode = "sequential"  # Default execution mode
        self.use_rl_decision_agent = use_rl_decision_agent  # Default to RL Decision Agent
        
        # Initialize input dependencies
        self.input_dependencies = input_dependencies if input_dependencies is not None else []
        
        # Initialize tools and dataloaders
        self.tools_dataloaders = tools_dataloaders if tools_dataloaders is not None else {}

    def perform_task(self, task, context, expected_outcome=None, human_feedback=6.5, input_quality=None):
        """Perform a given task, considering role optimization, learning, and tracking metrics."""
        self.history.append({"task": task, "context": context, "role": self.role})
        
        # Store human feedback score (default 6.5 if not provided)
        self.human_feedback_scores.append(human_feedback)
        
        # Store input quality assessment if provided
        if input_quality is not None:
            self.input_quality_scores.append(input_quality)
        
        # Store learning data if learning is enabled
        if self.enable_learning and expected_outcome is not None:
            self.learning_data.append((task, context, expected_outcome))
        
        # Adjust sequencing if enough data is collected and learning is enabled
        if self.enable_learning and len(self.learning_data) >= self.learning_threshold:
            self.optimize_workflow()
        
        return f"{self.name} executed {task} with role: {self.role}"
    
    def add_input_dependency(self, dependency):
        """Add a new input dependency for processing."""
        self.input_dependencies.append(dependency)
        self.history.append({
            "task": "Add input dependency",
            "context": f"Added new dependency: {type(dependency).__name__}",
            "role": self.role
        })
        return f"Added {type(dependency).__name__} as a dependency"
    
    def remove_input_dependency(self, dependency_index):
        """Remove an input dependency by index."""
        if 0 <= dependency_index < len(self.input_dependencies):
            dependency = self.input_dependencies.pop(dependency_index)
            self.history.append({
                "task": "Remove input dependency",
                "context": f"Removed dependency: {type(dependency).__name__}",
                "role": self.role
            })
            return f"Removed {type(dependency).__name__} dependency"
        return "Invalid dependency index"
    
    def register_tool(self, name, tool_function):
        """Register a new tool or dataloader."""
        self.tools_dataloaders[name] = tool_function
        self.history.append({
            "task": "Register tool",
            "context": f"Added new tool: {name}",
            "role": self.role
        })
        return f"Registered new tool: {name}"
    
    def unregister_tool(self, name):
        """Unregister a tool or dataloader by name."""
        if name in self.tools_dataloaders:
            del self.tools_dataloaders[name]
            self.history.append({
                "task": "Unregister tool",
                "context": f"Removed tool: {name}",
                "role": self.role
            })
            return f"Unregistered tool: {name}"
        return f"Tool {name} not found"
    
    def get_available_tools(self):
        """Get a list of all available tools."""
        return list(self.tools_dataloaders.keys())
    
    def use_tool(self, name, *args, **kwargs):
        """Use a registered tool with the given arguments."""
        if name not in self.tools_dataloaders:
            return f"Tool {name} not found. Available tools: {self.get_available_tools()}"
        
        tool = self.tools_dataloaders[name]
        self.history.append({
            "task": "Use tool",
            "context": f"Used tool: {name}",
            "role": self.role
        })
        
        try:
            return tool(*args, **kwargs)
        except Exception as e:
            return f"Error using tool {name}: {str(e)}"

    def adapt_role(self, new_role):
        """Allow external systems to update the agent's role dynamically."""
        self.role = new_role
        return f"{self.name} updated role to: {self.role}"

    def optimize_workflow(self):
        """Adjust the task execution sequence based on agent role, complexity, and quality at run-time."""
        if self.use_rl_decision_agent:
            self.optimize_with_rl_decision_agent()
        else:
            self.optimize_with_internal_rl()

    def optimize_with_rl_decision_agent(self):
        """Uses an external RL decision agent to determine workflow optimization."""
        print("Optimizing workflow using RL Decision Agent...")
        # Placeholder for RL integration logic

    def optimize_with_internal_rl(self):
        """Internal RL-based optimization using learning data."""
        success_rates = [1 if outcome == "success" else 0 for _, _, outcome in self.learning_data]
        avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
        avg_quality = self.average_input_quality() or 5  # Default quality if none available
        complexity_factor = len(self.skills)  # Higher complexity means more skilled agents
        
        # Decision logic: Adjust execution mode based on multiple factors
        if avg_success > 0.8 and avg_quality > 7 and complexity_factor > 3:
            self.execution_mode = "parallel"
        else:
            self.execution_mode = "sequential"
        
        print(f"Internal RL optimization complete: Execution mode set to {self.execution_mode}")

    def suggest_higher_input_quality(self):
        """Suggests increasing input data quality to improve accuracy based on feedback."""
        if not self.input_quality_scores:
            return "No input quality data available."
        avg_quality = sum(self.input_quality_scores) / len(self.input_quality_scores)
        suggested_quality = avg_quality * self.accuracy_boost_factor
        return f"Suggested minimum input quality: {suggested_quality:.2f} for better accuracy."

    def calculate_error_margin(self):
        """Calculate the inverse margin of error based on human feedback."""
        if not self.human_feedback_scores:
            return None
        avg_feedback = sum(self.human_feedback_scores) / len(self.human_feedback_scores)
        return 10 - avg_feedback  # Higher feedback = Lower error margin
    
    def average_input_quality(self):
        """Calculate the average quality of input data assessments."""
        if not self.input_quality_scores:
            return None
        return sum(self.input_quality_scores) / len(self.input_quality_scores)
    
    def add_custom_metric(self, metric_name, metric_function):
        """Allow users to define and add their own metric calculations."""
        self.custom_metrics[metric_name] = metric_function
    
    def evaluate_custom_metrics(self, *args, **kwargs):
        """Evaluate all user-defined metrics and return results."""
        results = {}
        for metric_name, metric_function in self.custom_metrics.items():
            results[metric_name] = metric_function(self, *args, **kwargs)
        return results

