from typing import Dict, List, Optional
import re

class TaskComplexityAnalyzer:
    """
    Analyzes the complexity of tasks to determine appropriate LLM parameters.
    This helps optimize model selection and generation parameters based on task needs.
    """
    
    # Complexity indicators (words that suggest complexity level)
    COMPLEXITY_INDICATORS = {
        "simple": [
            "brief", "quick", "simple", "short", "summarize", "list", "basic",
            "elementary", "straightforward", "easy", "concise", "outline"
        ],
        "medium": [
            "explain", "describe", "analyze", "compare", "contrast", "identify",
            "moderate", "standard", "typical", "regular", "normal", "general"
        ],
        "complex": [
            "comprehensive", "detailed", "in-depth", "thorough", "elaborate",
            "extensive", "evaluate", "assess", "synthesize", "complex", "intricate",
            "deep dive", "sophisticated", "nuanced"
        ],
        "creative": [
            "creative", "imagine", "innovative", "design", "generate", "invent",
            "story", "novel", "artistic", "unique", "original", "alternative",
            "fiction", "envision", "brainstorm"
        ]
    }
    
    # Role-based default complexity mappings
    ROLE_COMPLEXITY_MAPPING = {
        "researcher": "complex",
        "expert": "complex",
        "analyst": "complex",
        "writer": "medium",
        "assistant": "medium",
        "translator": "medium",
        "tutor": "complex",
        "summarizer": "simple",
        "coder": "complex",
        "code_assistant": "complex",
        "developer": "complex",
        "creative_writer": "creative",
        "content_creator": "creative",
        "extractor": "simple",
        "indexer": "simple",
        "analyzer": "medium",
        "validator": "medium",
        "verifier": "medium",
        "approver": "simple",
    }
    
    @classmethod
    def analyze_complexity(cls, text: str, role: Optional[str] = None) -> str:
        """
        Analyze the complexity of a task based on text content and role.
        
        Args:
            text: The task text to analyze
            role: Optional role for role-based complexity bias
            
        Returns:
            Complexity level: "simple", "medium", "complex", or "creative"
        """
        # Clean and normalize text
        text = text.lower().strip()
        
        # Count words to establish base complexity
        word_count = len(text.split())
        
        # Base complexity on word count
        base_complexity = "simple"
        if word_count > 50:
            base_complexity = "medium"
        if word_count > 200:
            base_complexity = "complex"
        
        # Check for complexity indicators in text
        indicator_scores = {
            "simple": 0,
            "medium": 0,
            "complex": 0,
            "creative": 0
        }
        
        # Count indicators for each complexity level
        for level, indicators in cls.COMPLEXITY_INDICATORS.items():
            for indicator in indicators:
                # Look for the indicator as a whole word
                matches = re.findall(r'\b' + re.escape(indicator) + r'\b', text)
                indicator_scores[level] += len(matches)
        
        # Determine the highest scoring complexity level
        max_score = 0
        content_complexity = base_complexity
        
        for level, score in indicator_scores.items():
            if score > max_score:
                max_score = score
                content_complexity = level
        
        # If no strong indicators found (max_score == 0), use base complexity
        if max_score == 0:
            content_complexity = base_complexity
            
        # Apply role-based bias if role is provided
        if role and role.lower() in cls.ROLE_COMPLEXITY_MAPPING:
            role_complexity = cls.ROLE_COMPLEXITY_MAPPING[role.lower()]
            
            # Compute final complexity, with content analysis having more weight
            # Content analysis: 70%, Role-based default: 30%
            if content_complexity == role_complexity:
                final_complexity = content_complexity
            elif max_score >= 3:  # Strong content indicators override role defaults
                final_complexity = content_complexity
            else:
                # Default to role complexity if content indicators aren't strong
                final_complexity = role_complexity
        else:
            final_complexity = content_complexity
        
        return final_complexity
    
    @classmethod
    def estimate_token_needs(cls, complexity: str, with_references: bool = False) -> int:
        """
        Estimate token needs based on complexity.
        
        Args:
            complexity: Task complexity ("simple", "medium", "complex", "creative")
            with_references: Whether references/citations are needed
            
        Returns:
            Estimated max tokens needed
        """
        # Base token estimates
        token_estimates = {
            "simple": 500,
            "medium": 1000,
            "complex": 2000,
            "creative": 1500
        }
        
        base_tokens = token_estimates.get(complexity, 1000)
        
        # Add extra tokens for references if needed
        if with_references:
            reference_tokens = base_tokens * 0.3  # 30% extra for references
            return int(base_tokens + reference_tokens)
        
        return base_tokens

# Example usage
if __name__ == "__main__":
    # Test with different texts
    test_texts = [
        "Briefly summarize the key points of the article.",
        "Explain the concept of neural networks in simple terms.",
        "Provide a comprehensive analysis of the economic factors that contributed to the 2008 financial crisis, including detailed examination of regulatory frameworks, market dynamics, and systemic risks. Include policy recommendations for preventing similar crises.",
        "Write a creative story about a time traveler who visits ancient Rome."
    ]
    
    for text in test_texts:
        complexity = TaskComplexityAnalyzer.analyze_complexity(text)
        tokens = TaskComplexityAnalyzer.estimate_token_needs(complexity)
        print(f"Text: {text[:50]}...")
        print(f"Complexity: {complexity}")
        print(f"Estimated tokens: {tokens}")
        print("-" * 50) 