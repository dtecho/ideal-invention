"""
Core DeepTreeEchoBot implementation with reinforcement learning capabilities.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from ..config.settings import BotConfig
from ..rl.agent import RLAgent
from ..search.agentic_search import AgenticSearchEngine
from ..reasoning.task_processor import TaskProcessor
from ..browsing.web_browser import WebBrowser
from ..utils.task_vectors import TaskVectorGenerator
from ..utils.reward_functions import SmoothRewardCalculator


class DeepTreeEchoBot:
    """
    Enhanced deep-tree-echo-bot with machine-generated task vectors,
    smooth reward functions, and pure reinforcement learning.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.rl_agent = RLAgent(config.rl_config)
        self.search_engine = AgenticSearchEngine(config.search_config)
        self.task_processor = TaskProcessor(config.reasoning_config)
        self.web_browser = WebBrowser(config.browsing_config)
        self.task_vector_gen = TaskVectorGenerator(config.task_vector_config)
        self.reward_calculator = SmoothRewardCalculator(config.reward_config)
        
        # State management
        self.current_state = None
        self.episode_history = []
        self.running = False
        
    async def initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing DeepTreeEchoBot...")
        
        await self.rl_agent.initialize()
        await self.search_engine.initialize()
        await self.task_processor.initialize()
        await self.web_browser.initialize()
        
        self.logger.info("DeepTreeEchoBot initialized successfully")
        
    async def process_task(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a task using reinforcement learning approach.
        
        Args:
            task: The task description
            context: Optional context information
            
        Returns:
            Dictionary containing results and metadata
        """
        start_time = datetime.now()
        
        # Generate task vector
        task_vector = await self.task_vector_gen.generate_vector(task, context)
        
        # Initialize episode
        episode_data = {
            'task': task,
            'context': context,
            'task_vector': task_vector,
            'start_time': start_time,
            'steps': [],
            'rewards': [],
            'total_reward': 0.0
        }
        
        # Main processing loop
        state = await self._get_initial_state(task, task_vector)
        done = False
        step = 0
        max_steps = self.config.max_steps_per_task
        
        while not done and step < max_steps:
            # Get action from RL agent
            action = await self.rl_agent.get_action(state, task_vector)
            
            # Execute action
            next_state, reward, done, info = await self._execute_action(
                action, state, task, task_vector
            )
            
            # Calculate smooth reward
            smooth_reward = self.reward_calculator.calculate_smooth_reward(
                reward, state, next_state, action, episode_data
            )
            
            # Store transition
            await self.rl_agent.store_transition(
                state, action, smooth_reward, next_state, done
            )
            
            # Update episode data
            episode_data['steps'].append({
                'step': step,
                'action': action,
                'reward': smooth_reward,
                'info': info
            })
            episode_data['rewards'].append(smooth_reward)
            episode_data['total_reward'] += smooth_reward
            
            state = next_state
            step += 1
            
        # Train the agent
        await self.rl_agent.train_step()
        
        # Finalize episode
        episode_data['end_time'] = datetime.now()
        episode_data['duration'] = (episode_data['end_time'] - start_time).total_seconds()
        episode_data['completed'] = done
        
        self.episode_history.append(episode_data)
        
        return {
            'success': done,
            'result': info.get('result', None) if 'info' in locals() else None,
            'total_reward': episode_data['total_reward'],
            'steps_taken': step,
            'duration': episode_data['duration'],
            'episode_data': episode_data
        }
        
    async def _get_initial_state(self, task: str, task_vector: np.ndarray) -> torch.Tensor:
        """Get initial state for the task."""
        # Combine task information with current environment state
        task_features = torch.tensor(task_vector, dtype=torch.float32)
        
        # Get search context
        search_context = await self.search_engine.get_context_features()
        
        # Get reasoning state
        reasoning_state = await self.task_processor.get_initial_state(task)
        
        # Ensure all tensors have compatible dimensions
        expected_total_dim = self.config.rl_config.state_dim
        
        # Pad or truncate to ensure correct dimensions
        if len(task_features) > expected_total_dim // 2:
            task_features = task_features[:expected_total_dim // 2]
        else:
            pad_size = expected_total_dim // 2 - len(task_features)
            task_features = torch.cat([task_features, torch.zeros(pad_size)])
        
        if len(search_context) > expected_total_dim // 4:
            search_context = search_context[:expected_total_dim // 4]
        else:
            pad_size = expected_total_dim // 4 - len(search_context)
            search_context = torch.cat([search_context, torch.zeros(pad_size)])
            
        if len(reasoning_state) > expected_total_dim // 4:
            reasoning_state = reasoning_state[:expected_total_dim // 4]
        else:
            pad_size = expected_total_dim // 4 - len(reasoning_state)
            reasoning_state = torch.cat([reasoning_state, torch.zeros(pad_size)])
        
        # Combine all features to exact state dimension
        state = torch.cat([task_features, search_context, reasoning_state])
        
        # Final padding if needed
        if len(state) < expected_total_dim:
            pad_size = expected_total_dim - len(state)
            state = torch.cat([state, torch.zeros(pad_size)])
        elif len(state) > expected_total_dim:
            state = state[:expected_total_dim]
        
        return state
        
    async def _execute_action(
        self, 
        action: int, 
        state: torch.Tensor, 
        task: str, 
        task_vector: np.ndarray
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute an action and return next state, reward, done flag, and info."""
        
        info = {'action_type': 'unknown', 'result': None}
        reward = 0.0
        done = False
        
        try:
            if action == 0:  # Search action
                search_result = await self.search_engine.search(
                    query=self._extract_search_query(task, state),
                    task_vector=task_vector
                )
                info.update({
                    'action_type': 'search',
                    'result': search_result,
                    'search_results': search_result.get('results', [])
                })
                reward = self._calculate_search_reward(search_result)
                
            elif action == 1:  # Browse action
                browse_result = await self.web_browser.browse(
                    url=self._extract_browse_url(task, state),
                    task_vector=task_vector
                )
                info.update({
                    'action_type': 'browse',
                    'result': browse_result,
                    'content': browse_result.get('content', '')
                })
                reward = self._calculate_browse_reward(browse_result)
                
            elif action == 2:  # Reasoning action
                reasoning_result = await self.task_processor.process_step(
                    task, state, task_vector
                )
                info.update({
                    'action_type': 'reasoning',
                    'result': reasoning_result,
                    'reasoning_output': reasoning_result.get('output', '')
                })
                reward = self._calculate_reasoning_reward(reasoning_result)
                
                # Check if task is completed
                done = reasoning_result.get('completed', False)
                
            elif action == 3:  # Synthesis action
                synthesis_result = await self._synthesize_information(state, task_vector)
                info.update({
                    'action_type': 'synthesis',
                    'result': synthesis_result,
                    'synthesized_result': synthesis_result.get('output', '')
                })
                reward = self._calculate_synthesis_reward(synthesis_result)
                done = synthesis_result.get('completed', False)
                
        except Exception as e:
            self.logger.error(f"Error executing action {action}: {e}")
            reward = -1.0
            info['error'] = str(e)
            
        # Get next state
        next_state = await self._get_next_state(state, action, info)
        
        return next_state, reward, done, info
        
    async def _get_next_state(
        self, 
        current_state: torch.Tensor, 
        action: int, 
        action_info: Dict
    ) -> torch.Tensor:
        """Compute next state based on current state and action results."""
        
        # Extract action result features
        action_features = torch.zeros(self.config.action_feature_dim)
        
        if action_info['action_type'] == 'search':
            # Encode search results
            search_results = action_info.get('search_results', [])
            if search_results:
                # Simple encoding: average of result relevance scores
                relevance_scores = [r.get('relevance', 0.0) for r in search_results]
                action_features[0] = np.mean(relevance_scores)
                action_features[1] = min(len(search_results) / 10.0, 1.0)  # Normalized count
                
        elif action_info['action_type'] == 'browse':
            # Encode browse results
            content = action_info.get('content', '')
            action_features[2] = min(len(content) / 1000.0, 1.0)  # Normalized content length
            action_features[3] = 1.0 if content else 0.0  # Success flag
            
        elif action_info['action_type'] == 'reasoning':
            # Encode reasoning results
            reasoning_output = action_info.get('reasoning_output', '')
            action_features[4] = min(len(reasoning_output) / 100.0, 1.0)  # Normalized output length
            action_features[5] = 1.0 if reasoning_output else 0.0  # Success flag
            
        elif action_info['action_type'] == 'synthesis':
            # Encode synthesis results
            synthesis_output = action_info.get('synthesized_result', '')
            action_features[6] = min(len(synthesis_output) / 100.0, 1.0)
            action_features[7] = 1.0 if synthesis_output else 0.0
            
        # Update state by replacing action features
        state_dim = len(current_state)
        action_dim = len(action_features)
        
        if state_dim >= action_dim:
            # Replace the last action_dim elements with new action features
            next_state = current_state.clone()
            next_state[-action_dim:] = action_features
        else:
            # If state is smaller than action features, just use action features
            next_state = action_features[:state_dim]
            
        return next_state
        
    def _extract_search_query(self, task: str, state: torch.Tensor) -> str:
        """Extract search query from task and state."""
        # Simple heuristic: use task as base query
        return task
        
    def _extract_browse_url(self, task: str, state: torch.Tensor) -> str:
        """Extract URL to browse from task and state."""
        # Simple heuristic: return a default search URL
        return f"https://www.google.com/search?q={task.replace(' ', '+')}"
        
    def _calculate_search_reward(self, search_result: Dict) -> float:
        """Calculate reward for search action."""
        results = search_result.get('results', [])
        if not results:
            return -0.1
            
        # Reward based on number and relevance of results
        relevance_sum = sum(r.get('relevance', 0.0) for r in results)
        return min(relevance_sum / len(results), 1.0)
        
    def _calculate_browse_reward(self, browse_result: Dict) -> float:
        """Calculate reward for browse action."""
        content = browse_result.get('content', '')
        if not content:
            return -0.1
            
        # Reward based on content length and quality
        content_score = min(len(content) / 1000.0, 1.0)
        return content_score * 0.5
        
    def _calculate_reasoning_reward(self, reasoning_result: Dict) -> float:
        """Calculate reward for reasoning action."""
        output = reasoning_result.get('output', '')
        completed = reasoning_result.get('completed', False)
        
        base_reward = 0.3 if output else -0.1
        completion_bonus = 1.0 if completed else 0.0
        
        return base_reward + completion_bonus
        
    def _calculate_synthesis_reward(self, synthesis_result: Dict) -> float:
        """Calculate reward for synthesis action."""
        output = synthesis_result.get('output', '')
        completed = synthesis_result.get('completed', False)
        
        base_reward = 0.5 if output else -0.1
        completion_bonus = 2.0 if completed else 0.0
        
        return base_reward + completion_bonus
        
    async def _synthesize_information(
        self, 
        state: torch.Tensor, 
        task_vector: np.ndarray
    ) -> Dict:
        """Synthesize information from current state to produce final result."""
        
        # Simple synthesis: combine available information
        synthesis_output = "Task processing completed based on available information."
        
        return {
            'output': synthesis_output,
            'completed': True,
            'confidence': 0.8
        }
        
    async def shutdown(self):
        """Shutdown all components gracefully."""
        self.logger.info("Shutting down DeepTreeEchoBot...")
        self.running = False
        
        await self.web_browser.close()
        await self.search_engine.shutdown()
        
        self.logger.info("DeepTreeEchoBot shutdown complete")