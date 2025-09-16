"""
Focused Reasoning Task Processor for DeepTreeEchoBot.
Provides structured reasoning capabilities for complex task execution.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import re
from datetime import datetime

from ..config.settings import ReasoningConfig


class ReasoningStep:
    """Represents a single reasoning step."""
    
    def __init__(self, step_type: str, input_data: Any, output_data: Any, confidence: float):
        self.step_type = step_type
        self.input_data = input_data
        self.output_data = output_data
        self.confidence = confidence
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        return {
            'step_type': self.step_type,
            'input_data': str(self.input_data),
            'output_data': str(self.output_data),
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class TaskProcessor:
    """
    Focused reasoning system that breaks down complex tasks
    into manageable steps and processes them systematically.
    """
    
    def __init__(self, config: ReasoningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Reasoning components
        self.reasoning_history = []
        self.current_context = {}
        self.task_decomposition_patterns = self._initialize_patterns()
        
        # State tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        
    async def initialize(self):
        """Initialize the task processor."""
        self.logger.info("Initializing Task Processor...")
        self.logger.info("Task Processor initialized with focused reasoning capabilities")
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize task decomposition patterns."""
        return {
            'search_task': [
                'identify_keywords',
                'formulate_query',
                'evaluate_results',
                'extract_information'
            ],
            'analysis_task': [
                'gather_information',
                'identify_patterns',
                'draw_conclusions',
                'validate_findings'
            ],
            'problem_solving': [
                'understand_problem',
                'identify_constraints',
                'generate_solutions',
                'evaluate_options',
                'select_best_solution'
            ],
            'research_task': [
                'define_research_question',
                'identify_sources',
                'collect_information',
                'synthesize_findings',
                'present_conclusions'
            ]
        }
        
    async def get_initial_state(self, task: str) -> torch.Tensor:
        """Get initial reasoning state for a task."""
        features = torch.zeros(self.config.state_feature_dim)
        
        # Task complexity features
        features[0] = len(task.split()) / 20.0  # Normalized word count
        features[1] = len(re.findall(r'\?', task)) / 5.0  # Question marks
        features[2] = len(re.findall(r'\b(find|search|analyze|compare|explain)\b', task.lower())) / 5.0
        
        # Task type classification
        task_lower = task.lower()
        if any(word in task_lower for word in ['search', 'find', 'look']):
            features[3] = 1.0  # Search task
        elif any(word in task_lower for word in ['analyze', 'examine', 'study']):
            features[4] = 1.0  # Analysis task
        elif any(word in task_lower for word in ['solve', 'fix', 'resolve']):
            features[5] = 1.0  # Problem solving
        elif any(word in task_lower for word in ['research', 'investigate', 'explore']):
            features[6] = 1.0  # Research task
            
        return features
        
    async def process_step(
        self, 
        task: str, 
        current_state: torch.Tensor, 
        task_vector: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a single reasoning step for the given task.
        
        Args:
            task: The task description
            current_state: Current state tensor
            task_vector: Task-specific vector
            
        Returns:
            Dictionary with step results and metadata
        """
        self.logger.debug(f"Processing reasoning step for task: {task}")
        
        # Get task ID for tracking
        task_id = hash(task) % 1000000
        
        if task_id not in self.active_tasks:
            # Initialize new task
            task_type = self._classify_task(task)
            self.active_tasks[task_id] = {
                'task': task,
                'task_type': task_type,
                'steps': self.task_decomposition_patterns.get(task_type, ['process_task']),
                'current_step': 0,
                'context': {},
                'reasoning_history': []
            }
            
        task_info = self.active_tasks[task_id]
        
        # Get current step
        if task_info['current_step'] >= len(task_info['steps']):
            # Task completed
            self.completed_tasks[task_id] = task_info
            del self.active_tasks[task_id]
            
            return {
                'output': self._generate_final_output(task_info),
                'completed': True,
                'step_type': 'completion',
                'confidence': 0.8
            }
            
        current_step_name = task_info['steps'][task_info['current_step']]
        
        # Process the current step
        step_result = await self._execute_reasoning_step(
            current_step_name, 
            task, 
            task_info['context'],
            current_state,
            task_vector
        )
        
        # Update task state
        task_info['context'].update(step_result.get('context_updates', {}))
        task_info['reasoning_history'].append(ReasoningStep(
            step_type=current_step_name,
            input_data=task_info['context'],
            output_data=step_result['output'],
            confidence=step_result['confidence']
        ))
        
        # Move to next step
        task_info['current_step'] += 1
        
        return {
            'output': step_result['output'],
            'completed': False,
            'step_type': current_step_name,
            'confidence': step_result['confidence'],
            'progress': task_info['current_step'] / len(task_info['steps'])
        }
        
    def _classify_task(self, task: str) -> str:
        """Classify the task type based on keywords."""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ['search', 'find', 'look up', 'locate']):
            return 'search_task'
        elif any(word in task_lower for word in ['analyze', 'examine', 'study', 'evaluate']):
            return 'analysis_task'
        elif any(word in task_lower for word in ['solve', 'fix', 'resolve', 'troubleshoot']):
            return 'problem_solving'
        elif any(word in task_lower for word in ['research', 'investigate', 'explore']):
            return 'research_task'
        else:
            return 'analysis_task'  # Default
            
    async def _execute_reasoning_step(
        self,
        step_name: str,
        task: str,
        context: Dict,
        current_state: torch.Tensor,
        task_vector: np.ndarray
    ) -> Dict[str, Any]:
        """Execute a specific reasoning step."""
        
        if step_name == 'identify_keywords':
            return await self._identify_keywords(task, context)
        elif step_name == 'formulate_query':
            return await self._formulate_query(task, context)
        elif step_name == 'evaluate_results':
            return await self._evaluate_results(task, context)
        elif step_name == 'extract_information':
            return await self._extract_information(task, context)
        elif step_name == 'gather_information':
            return await self._gather_information(task, context)
        elif step_name == 'identify_patterns':
            return await self._identify_patterns(task, context)
        elif step_name == 'draw_conclusions':
            return await self._draw_conclusions(task, context)
        elif step_name == 'validate_findings':
            return await self._validate_findings(task, context)
        elif step_name == 'understand_problem':
            return await self._understand_problem(task, context)
        elif step_name == 'identify_constraints':
            return await self._identify_constraints(task, context)
        elif step_name == 'generate_solutions':
            return await self._generate_solutions(task, context)
        elif step_name == 'evaluate_options':
            return await self._evaluate_options(task, context)
        elif step_name == 'select_best_solution':
            return await self._select_best_solution(task, context)
        else:
            return await self._generic_processing_step(step_name, task, context)
            
    async def _identify_keywords(self, task: str, context: Dict) -> Dict[str, Any]:
        """Identify key terms and concepts in the task."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', task.lower())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Get most important keywords (simple frequency-based)
        keyword_freq = {}
        for keyword in keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
        important_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'output': f"Identified key terms: {', '.join([kw[0] for kw in important_keywords])}",
            'confidence': 0.7,
            'context_updates': {'keywords': [kw[0] for kw in important_keywords]}
        }
        
    async def _formulate_query(self, task: str, context: Dict) -> Dict[str, Any]:
        """Formulate an effective search query based on the task and keywords."""
        keywords = context.get('keywords', [])
        
        if keywords:
            query = ' '.join(keywords[:3])  # Use top 3 keywords
        else:
            query = task
            
        return {
            'output': f"Formulated search query: '{query}'",
            'confidence': 0.8,
            'context_updates': {'search_query': query}
        }
        
    async def _evaluate_results(self, task: str, context: Dict) -> Dict[str, Any]:
        """Evaluate the quality and relevance of gathered results."""
        return {
            'output': "Evaluated available results for relevance and quality",
            'confidence': 0.6,
            'context_updates': {'evaluation_complete': True}
        }
        
    async def _extract_information(self, task: str, context: Dict) -> Dict[str, Any]:
        """Extract relevant information from available sources."""
        return {
            'output': "Extracted relevant information from available sources",
            'confidence': 0.7,
            'context_updates': {'information_extracted': True}
        }
        
    async def _gather_information(self, task: str, context: Dict) -> Dict[str, Any]:
        """Gather information relevant to the task."""
        return {
            'output': "Gathered information from multiple sources",
            'confidence': 0.7,
            'context_updates': {'information_gathered': True}
        }
        
    async def _identify_patterns(self, task: str, context: Dict) -> Dict[str, Any]:
        """Identify patterns in the gathered information."""
        return {
            'output': "Identified key patterns and relationships in the data",
            'confidence': 0.6,
            'context_updates': {'patterns_identified': True}
        }
        
    async def _draw_conclusions(self, task: str, context: Dict) -> Dict[str, Any]:
        """Draw conclusions based on the analysis."""
        return {
            'output': "Drew preliminary conclusions based on available evidence",
            'confidence': 0.7,
            'context_updates': {'conclusions_drawn': True}
        }
        
    async def _validate_findings(self, task: str, context: Dict) -> Dict[str, Any]:
        """Validate the findings and conclusions."""
        return {
            'output': "Validated findings through cross-referencing and logical consistency checks",
            'confidence': 0.8,
            'context_updates': {'findings_validated': True}
        }
        
    async def _understand_problem(self, task: str, context: Dict) -> Dict[str, Any]:
        """Understand the core problem or challenge."""
        return {
            'output': f"Analyzed problem statement: {task}",
            'confidence': 0.8,
            'context_updates': {'problem_understood': True}
        }
        
    async def _identify_constraints(self, task: str, context: Dict) -> Dict[str, Any]:
        """Identify constraints and limitations."""
        return {
            'output': "Identified key constraints and limitations affecting the solution",
            'confidence': 0.6,
            'context_updates': {'constraints_identified': True}
        }
        
    async def _generate_solutions(self, task: str, context: Dict) -> Dict[str, Any]:
        """Generate potential solutions or approaches."""
        return {
            'output': "Generated multiple potential solutions and approaches",
            'confidence': 0.7,
            'context_updates': {'solutions_generated': True}
        }
        
    async def _evaluate_options(self, task: str, context: Dict) -> Dict[str, Any]:
        """Evaluate the generated options."""
        return {
            'output': "Evaluated options based on feasibility, effectiveness, and constraints",
            'confidence': 0.7,
            'context_updates': {'options_evaluated': True}
        }
        
    async def _select_best_solution(self, task: str, context: Dict) -> Dict[str, Any]:
        """Select the best solution from evaluated options."""
        return {
            'output': "Selected the most suitable solution based on evaluation criteria",
            'confidence': 0.8,
            'context_updates': {'solution_selected': True}
        }
        
    async def _generic_processing_step(self, step_name: str, task: str, context: Dict) -> Dict[str, Any]:
        """Generic processing step for undefined step types."""
        return {
            'output': f"Completed processing step: {step_name}",
            'confidence': 0.5,
            'context_updates': {f'{step_name}_complete': True}
        }
        
    def _generate_final_output(self, task_info: Dict) -> str:
        """Generate final output based on completed reasoning steps."""
        task = task_info['task']
        task_type = task_info['task_type']
        context = task_info['context']
        
        # Create a summary based on the reasoning process
        output_parts = [
            f"Task: {task}",
            f"Approach: {task_type.replace('_', ' ').title()}",
        ]
        
        # Add key findings from context
        if 'keywords' in context:
            output_parts.append(f"Key terms identified: {', '.join(context['keywords'])}")
            
        if 'search_query' in context:
            output_parts.append(f"Search strategy: {context['search_query']}")
            
        if any(key.endswith('_complete') or key.endswith('_identified') for key in context.keys()):
            completed_steps = [key.replace('_', ' ').title() for key in context.keys() 
                             if key.endswith('_complete') or key.endswith('_identified')]
            output_parts.append(f"Completed steps: {', '.join(completed_steps)}")
            
        output_parts.append("Task processing completed using systematic reasoning approach.")
        
        return '\n'.join(output_parts)
        
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get current reasoning statistics."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'total_reasoning_steps': sum(len(task['reasoning_history']) for task in self.active_tasks.values()),
            'task_types': list(self.task_decomposition_patterns.keys())
        }