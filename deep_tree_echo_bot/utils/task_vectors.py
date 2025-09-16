"""
Machine-Generated Task Vectors for DeepTreeEchoBot.
Creates task-specific vector representations for optimized thinking processes.
"""
import logging
import numpy as np
import re
from typing import Dict, List, Optional, Any
import hashlib
from datetime import datetime

from ..config.settings import TaskVectorConfig


class TaskVectorGenerator:
    """
    Generates machine-learned task vectors that optimize thinking processes
    and guide the bot's decision-making for specific task types.
    """
    
    def __init__(self, config: TaskVectorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Task type templates
        self.task_templates = self._initialize_task_templates()
        
        # Learned vector patterns
        self.vector_patterns = {}
        self.task_history = []
        
    def _initialize_task_templates(self) -> Dict[str, np.ndarray]:
        """Initialize base task vector templates."""
        dim = self.config.vector_dimension
        
        templates = {
            'search': self._create_search_template(dim),
            'analysis': self._create_analysis_template(dim),
            'problem_solving': self._create_problem_solving_template(dim),
            'research': self._create_research_template(dim),
            'synthesis': self._create_synthesis_template(dim),
            'comparison': self._create_comparison_template(dim),
            'explanation': self._create_explanation_template(dim)
        }
        
        return templates
        
    def _create_search_template(self, dim: int) -> np.ndarray:
        """Create template for search tasks."""
        template = np.zeros(dim)
        
        # Search-specific features
        template[0] = 1.0   # Primary search indicator
        template[1] = 0.8   # Information gathering weight
        template[2] = 0.6   # Exploration emphasis
        template[3] = 0.7   # Breadth over depth
        template[4] = 0.5   # Speed emphasis
        
        # Add some learned randomness for exploration
        template[5:15] = np.random.normal(0, 0.1, 10)
        
        return template
        
    def _create_analysis_template(self, dim: int) -> np.ndarray:
        """Create template for analysis tasks."""
        template = np.zeros(dim)
        
        # Analysis-specific features
        template[0] = 0.3   # Search component
        template[1] = 1.0   # Primary analysis indicator
        template[2] = 0.8   # Deep processing weight
        template[3] = 0.4   # Depth over breadth
        template[4] = 0.3   # Thoroughness over speed
        template[5] = 0.9   # Pattern recognition
        template[6] = 0.7   # Critical thinking
        
        # Add learned patterns
        template[7:17] = np.random.normal(0, 0.1, 10)
        
        return template
        
    def _create_problem_solving_template(self, dim: int) -> np.ndarray:
        """Create template for problem-solving tasks."""
        template = np.zeros(dim)
        
        # Problem-solving features
        template[0] = 0.5   # Search component
        template[1] = 0.7   # Analysis component
        template[2] = 1.0   # Primary problem-solving indicator
        template[3] = 0.8   # Solution generation
        template[4] = 0.6   # Constraint handling
        template[5] = 0.9   # Creative thinking
        template[6] = 0.7   # Evaluation emphasis
        
        # Add learned patterns
        template[7:17] = np.random.normal(0, 0.15, 10)
        
        return template
        
    def _create_research_template(self, dim: int) -> np.ndarray:
        """Create template for research tasks."""
        template = np.zeros(dim)
        
        # Research-specific features
        template[0] = 0.9   # High search component
        template[1] = 0.8   # Analysis component
        template[2] = 0.6   # Problem-solving component
        template[3] = 1.0   # Primary research indicator
        template[4] = 0.8   # Systematic approach
        template[5] = 0.7   # Source verification
        template[6] = 0.6   # Comprehensive coverage
        
        # Add learned patterns
        template[7:17] = np.random.normal(0, 0.12, 10)
        
        return template
        
    def _create_synthesis_template(self, dim: int) -> np.ndarray:
        """Create template for synthesis tasks."""
        template = np.zeros(dim)
        
        # Synthesis features
        template[0] = 0.4   # Search component
        template[1] = 0.9   # Analysis component
        template[2] = 0.7   # Problem-solving component
        template[3] = 0.8   # Research component
        template[4] = 1.0   # Primary synthesis indicator
        template[5] = 0.9   # Integration ability
        template[6] = 0.8   # Summary generation
        
        # Add learned patterns
        template[7:17] = np.random.normal(0, 0.1, 10)
        
        return template
        
    def _create_comparison_template(self, dim: int) -> np.ndarray:
        """Create template for comparison tasks."""
        template = np.zeros(dim)
        
        # Comparison features
        template[0] = 0.6   # Search component
        template[1] = 1.0   # High analysis component
        template[2] = 0.5   # Problem-solving component
        template[3] = 0.7   # Research component
        template[4] = 0.8   # Synthesis component
        template[5] = 0.9   # Primary comparison indicator
        template[6] = 0.8   # Evaluation emphasis
        
        # Add learned patterns
        template[7:17] = np.random.normal(0, 0.1, 10)
        
        return template
        
    def _create_explanation_template(self, dim: int) -> np.ndarray:
        """Create template for explanation tasks."""
        template = np.zeros(dim)
        
        # Explanation features
        template[0] = 0.5   # Search component
        template[1] = 0.8   # Analysis component
        template[2] = 0.6   # Problem-solving component
        template[3] = 0.7   # Research component
        template[4] = 0.9   # Synthesis component
        template[5] = 0.7   # Comparison component
        template[6] = 1.0   # Primary explanation indicator
        template[7] = 0.8   # Clarity emphasis
        template[8] = 0.7   # Step-by-step approach
        
        # Add learned patterns
        template[9:19] = np.random.normal(0, 0.1, 10)
        
        return template
        
    async def generate_vector(self, task: str, context: Optional[Dict] = None) -> np.ndarray:
        """
        Generate a machine-learned task vector for the given task.
        
        Args:
            task: Task description
            context: Optional context information
            
        Returns:
            Task vector as numpy array
        """
        self.logger.debug(f"Generating task vector for: {task}")
        
        # Classify task type
        task_type = self._classify_task_type(task)
        
        # Get base template
        base_vector = self.task_templates[task_type].copy()
        
        # Apply task-specific modifications
        task_vector = await self._customize_vector(base_vector, task, context, task_type)
        
        # Apply learned optimizations
        optimized_vector = self._apply_learned_optimizations(task_vector, task, task_type)
        
        # Store for learning
        self._store_task_vector(task, task_type, optimized_vector, context)
        
        return optimized_vector
        
    def _classify_task_type(self, task: str) -> str:
        """Classify the task type based on keywords and patterns."""
        task_lower = task.lower()
        
        # Define keyword patterns for each task type
        patterns = {
            'search': [
                r'\b(find|search|look|locate|discover)\b',
                r'\b(what is|where is|who is|when is)\b',
                r'\b(information about|details about|data on)\b'
            ],
            'analysis': [
                r'\b(analyze|examine|study|evaluate|assess)\b',
                r'\b(breakdown|dissect|investigate|inspect)\b',
                r'\b(patterns|trends|characteristics)\b'
            ],
            'problem_solving': [
                r'\b(solve|fix|resolve|address|handle)\b',
                r'\b(problem|issue|challenge|difficulty)\b',
                r'\b(solution|approach|method|strategy)\b'
            ],
            'research': [
                r'\b(research|investigate|explore|study)\b',
                r'\b(comprehensive|thorough|detailed|extensive)\b',
                r'\b(sources|references|literature|evidence)\b'
            ],
            'synthesis': [
                r'\b(summarize|combine|integrate|merge)\b',
                r'\b(overall|general|comprehensive|unified)\b',
                r'\b(conclusion|summary|synthesis|overview)\b'
            ],
            'comparison': [
                r'\b(compare|contrast|difference|similarity)\b',
                r'\b(versus|vs|between|among)\b',
                r'\b(better|worse|advantage|disadvantage)\b'
            ],
            'explanation': [
                r'\b(explain|describe|define|clarify)\b',
                r'\b(how|why|what)\b',
                r'\b(meaning|definition|concept|idea)\b'
            ]
        }
        
        # Score each task type
        scores = {}
        for task_type, pattern_list in patterns.items():
            score = 0
            for pattern in pattern_list:
                matches = len(re.findall(pattern, task_lower))
                score += matches
            scores[task_type] = score
            
        # Return the highest scoring task type
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
                
        # Default to analysis if no clear match
        return 'analysis'
        
    async def _customize_vector(
        self, 
        base_vector: np.ndarray, 
        task: str, 
        context: Optional[Dict], 
        task_type: str
    ) -> np.ndarray:
        """Customize the base vector for the specific task."""
        vector = base_vector.copy()
        
        # Task complexity adjustment
        complexity = self._calculate_task_complexity(task)
        vector = vector * (1.0 + complexity * 0.1)
        
        # Context-based adjustments
        if context:
            # Urgency adjustment
            if context.get('urgent', False):
                vector[4] = min(1.0, vector[4] + 0.2)  # Increase speed emphasis
                
            # Quality emphasis adjustment
            if context.get('high_quality', False):
                vector[2] = min(1.0, vector[2] + 0.2)  # Increase thoroughness
                
            # Domain-specific adjustments
            domain = context.get('domain', '')
            if domain:
                domain_adjustment = self._get_domain_adjustment(domain)
                vector[:len(domain_adjustment)] += domain_adjustment
                
        # Task-specific fine-tuning
        if 'creative' in task.lower() or 'innovative' in task.lower():
            if len(vector) > 10:
                vector[10] = min(1.0, vector[10] + 0.3)  # Creativity boost
                
        if 'accurate' in task.lower() or 'precise' in task.lower():
            if len(vector) > 11:
                vector[11] = min(1.0, vector[11] + 0.3)  # Precision boost
                
        # Normalize to prevent overflow
        vector = np.clip(vector, -1.0, 1.0)
        
        return vector
        
    def _calculate_task_complexity(self, task: str) -> float:
        """Calculate task complexity score."""
        # Simple heuristics for complexity
        complexity = 0.0
        
        # Length factor
        complexity += min(len(task) / 200.0, 0.5)
        
        # Question count
        complexity += len(re.findall(r'\?', task)) * 0.1
        
        # Complex keywords
        complex_keywords = ['analyze', 'compare', 'evaluate', 'synthesize', 'comprehensive']
        for keyword in complex_keywords:
            if keyword in task.lower():
                complexity += 0.2
                
        # Multiple requirements
        if len(re.findall(r'\band\b|\bor\b|\balso\b|\badditionally\b', task.lower())) > 0:
            complexity += 0.2
            
        return min(complexity, 1.0)
        
    def _get_domain_adjustment(self, domain: str) -> np.ndarray:
        """Get domain-specific vector adjustments."""
        adjustments = {
            'technical': np.array([0.1, 0.2, 0.1, 0.0, -0.1]),
            'creative': np.array([-0.1, 0.0, 0.3, 0.2, 0.1]),
            'business': np.array([0.2, 0.1, -0.1, 0.1, 0.2]),
            'academic': np.array([0.0, 0.3, 0.2, 0.2, -0.2]),
            'scientific': np.array([0.1, 0.3, 0.1, 0.3, -0.1])
        }
        
        return adjustments.get(domain.lower(), np.zeros(5))
        
    def _apply_learned_optimizations(
        self, 
        vector: np.ndarray, 
        task: str, 
        task_type: str
    ) -> np.ndarray:
        """Apply learned optimizations based on task performance history."""
        # Generate task signature for learning
        task_signature = self._generate_task_signature(task, task_type)
        
        if task_signature in self.vector_patterns:
            # Apply learned optimizations
            learned_pattern = self.vector_patterns[task_signature]
            
            # Blend with learned pattern (weighted average)
            blend_weight = self.config.learning_rate
            vector = (1 - blend_weight) * vector + blend_weight * learned_pattern['vector']
            
            self.logger.debug(f"Applied learned optimization for task type: {task_type}")
            
        return vector
        
    def _generate_task_signature(self, task: str, task_type: str) -> str:
        """Generate a signature for task pattern matching."""
        # Create a hash based on task type and key features
        key_features = []
        
        # Add task type
        key_features.append(task_type)
        
        # Add key words (normalized)
        words = re.findall(r'\b\w+\b', task.lower())
        important_words = [w for w in words if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'they', 'have', 'will']]
        key_features.extend(sorted(important_words[:5]))  # Top 5 important words
        
        # Add task length category
        if len(task) < 50:
            key_features.append('short')
        elif len(task) < 150:
            key_features.append('medium')
        else:
            key_features.append('long')
            
        # Create hash
        signature_string = '|'.join(key_features)
        return hashlib.md5(signature_string.encode()).hexdigest()[:16]
        
    def _store_task_vector(
        self, 
        task: str, 
        task_type: str, 
        vector: np.ndarray, 
        context: Optional[Dict]
    ):
        """Store task vector for learning purposes."""
        task_data = {
            'task': task[:100],  # Truncate for storage
            'task_type': task_type,
            'vector': vector.copy(),
            'context': context,
            'timestamp': datetime.now()
        }
        
        self.task_history.append(task_data)
        
        # Keep only recent history (memory management)
        if len(self.task_history) > self.config.max_history_size:
            self.task_history = self.task_history[-self.config.max_history_size:]
            
    def update_pattern_from_feedback(
        self, 
        task: str, 
        task_type: str, 
        performance_score: float
    ):
        """Update learned patterns based on performance feedback."""
        task_signature = self._generate_task_signature(task, task_type)
        
        # Find corresponding vector from history
        for task_data in reversed(self.task_history):
            if (task_data['task_type'] == task_type and 
                self._generate_task_signature(task_data['task'], task_type) == task_signature):
                
                if task_signature not in self.vector_patterns:
                    self.vector_patterns[task_signature] = {
                        'vector': task_data['vector'].copy(),
                        'performance_history': [],
                        'update_count': 0
                    }
                    
                pattern = self.vector_patterns[task_signature]
                pattern['performance_history'].append(performance_score)
                pattern['update_count'] += 1
                
                # Update vector based on performance
                if performance_score > 0.7:  # Good performance
                    # Reinforce current pattern
                    pattern['vector'] = 0.9 * pattern['vector'] + 0.1 * task_data['vector']
                elif performance_score < 0.3:  # Poor performance
                    # Add exploration noise
                    noise = np.random.normal(0, 0.1, len(pattern['vector']))
                    pattern['vector'] = pattern['vector'] + noise
                    pattern['vector'] = np.clip(pattern['vector'], -1.0, 1.0)
                    
                self.logger.debug(f"Updated pattern for {task_signature} based on performance {performance_score}")
                break
                
    def get_vector_stats(self) -> Dict[str, Any]:
        """Get statistics about task vectors."""
        return {
            'total_tasks_processed': len(self.task_history),
            'learned_patterns': len(self.vector_patterns),
            'task_types': list(self.task_templates.keys()),
            'vector_dimension': self.config.vector_dimension,
            'recent_task_types': [t['task_type'] for t in self.task_history[-10:]]
        }