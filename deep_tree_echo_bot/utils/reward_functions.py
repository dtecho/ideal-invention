"""
Smooth Reward Functions for DeepTreeEchoBot.
Implements sophisticated reward calculation across multiple categories
for pure reinforcement learning optimization.
"""
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import math

from ..config.settings import RewardConfig


class RewardCategory:
    """Represents a reward category with its own calculation logic."""
    
    def __init__(self, name: str, weight: float, smoothing_factor: float = 0.1):
        self.name = name
        self.weight = weight
        self.smoothing_factor = smoothing_factor
        self.history = []
        self.moving_average = 0.0
        
    def update(self, reward: float):
        """Update the category with a new reward value."""
        self.history.append({
            'reward': reward,
            'timestamp': datetime.now()
        })
        
        # Update moving average
        if self.moving_average == 0.0:
            self.moving_average = reward
        else:
            self.moving_average = (
                (1 - self.smoothing_factor) * self.moving_average + 
                self.smoothing_factor * reward
            )
            
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.history = [h for h in self.history if h['timestamp'] > cutoff_time]
        
    def get_smoothed_reward(self) -> float:
        """Get the smoothed reward value."""
        return self.moving_average
        
    def get_recent_variance(self) -> float:
        """Calculate variance in recent rewards."""
        if len(self.history) < 2:
            return 0.0
            
        recent_rewards = [h['reward'] for h in self.history[-10:]]
        return np.var(recent_rewards)


class SmoothRewardCalculator:
    """
    Advanced reward calculator that provides smooth, multi-category rewards
    for stable reinforcement learning training.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize reward categories
        self.categories = self._initialize_categories()
        
        # Global reward state
        self.global_reward_history = []
        self.episode_rewards = []
        
        # Adaptive parameters
        self.exploration_bonus_decay = 0.99
        self.current_exploration_bonus = config.initial_exploration_bonus
        
    def _initialize_categories(self) -> Dict[str, RewardCategory]:
        """Initialize reward categories with weights."""
        categories = {
            'task_completion': RewardCategory('task_completion', 0.25, 0.15),
            'efficiency': RewardCategory('efficiency', 0.20, 0.10),
            'quality': RewardCategory('quality', 0.20, 0.12),
            'exploration': RewardCategory('exploration', 0.15, 0.08),
            'learning': RewardCategory('learning', 0.10, 0.05),
            'coherence': RewardCategory('coherence', 0.10, 0.07)
        }
        
        self.logger.info(f"Initialized {len(categories)} reward categories")
        return categories
        
    def calculate_smooth_reward(
        self,
        base_reward: float,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: int,
        episode_data: Dict[str, Any]
    ) -> float:
        """
        Calculate smooth reward across multiple categories.
        
        Args:
            base_reward: Raw reward from action execution
            state: Current state tensor
            next_state: Next state tensor  
            action: Action taken
            episode_data: Current episode information
            
        Returns:
            Smoothed multi-category reward
        """
        # Calculate rewards for each category
        category_rewards = {}
        
        # Task completion reward
        category_rewards['task_completion'] = self._calculate_completion_reward(
            base_reward, episode_data
        )
        
        # Efficiency reward
        category_rewards['efficiency'] = self._calculate_efficiency_reward(
            state, next_state, action, episode_data
        )
        
        # Quality reward
        category_rewards['quality'] = self._calculate_quality_reward(
            base_reward, state, next_state, episode_data
        )
        
        # Exploration reward
        category_rewards['exploration'] = self._calculate_exploration_reward(
            action, state, episode_data
        )
        
        # Learning reward
        category_rewards['learning'] = self._calculate_learning_reward(
            state, next_state, episode_data
        )
        
        # Coherence reward
        category_rewards['coherence'] = self._calculate_coherence_reward(
            action, episode_data
        )
        
        # Update categories and calculate weighted sum
        total_reward = 0.0
        for category_name, reward in category_rewards.items():
            category = self.categories[category_name]
            category.update(reward)
            
            # Use smoothed reward for calculation
            smoothed_reward = category.get_smoothed_reward()
            weighted_reward = smoothed_reward * category.weight
            total_reward += weighted_reward
            
        # Apply global smoothing
        smooth_reward = self._apply_global_smoothing(total_reward, base_reward)
        
        # Add exploration bonus (decaying)
        exploration_bonus = self._calculate_exploration_bonus(action, episode_data)
        smooth_reward += exploration_bonus
        
        # Apply temporal smoothing
        temporal_smooth_reward = self._apply_temporal_smoothing(smooth_reward)
        
        # Store for history
        self.global_reward_history.append({
            'reward': temporal_smooth_reward,
            'category_rewards': category_rewards,
            'base_reward': base_reward,
            'timestamp': datetime.now()
        })
        
        # Limit history size
        if len(self.global_reward_history) > self.config.max_history_size:
            self.global_reward_history = self.global_reward_history[-self.config.max_history_size:]
            
        self.logger.debug(f"Calculated smooth reward: {temporal_smooth_reward:.4f} (base: {base_reward:.4f})")
        
        return temporal_smooth_reward
        
    def _calculate_completion_reward(self, base_reward: float, episode_data: Dict) -> float:
        """Calculate task completion reward."""
        steps = episode_data.get('steps', [])
        
        if not steps:
            return base_reward * 0.5
            
        # Progress-based reward
        progress_reward = len(steps) * 0.1
        
        # Success bonus
        success_bonus = 1.0 if base_reward > 0.5 else 0.0
        
        # Time penalty (slight)
        time_penalty = min(len(steps) * 0.02, 0.3)
        
        completion_reward = base_reward + progress_reward + success_bonus - time_penalty
        
        return np.clip(completion_reward, -1.0, 2.0)
        
    def _calculate_efficiency_reward(
        self, 
        state: torch.Tensor, 
        next_state: torch.Tensor, 
        action: int, 
        episode_data: Dict
    ) -> float:
        """Calculate efficiency reward based on action effectiveness."""
        steps = episode_data.get('steps', [])
        
        # Base efficiency from state change
        state_change = torch.norm(next_state - state).item()
        efficiency_base = min(state_change / 10.0, 1.0)  # Normalize
        
        # Step efficiency (fewer steps is better for simple tasks)
        step_count = len(steps)
        step_efficiency = max(0.0, 1.0 - step_count * 0.1)
        
        # Action repetition penalty
        recent_actions = [s.get('action', -1) for s in steps[-5:]]
        repetition_penalty = recent_actions.count(action) * 0.1
        
        efficiency_reward = efficiency_base + step_efficiency - repetition_penalty
        
        return np.clip(efficiency_reward, -0.5, 1.0)
        
    def _calculate_quality_reward(
        self, 
        base_reward: float, 
        state: torch.Tensor, 
        next_state: torch.Tensor, 
        episode_data: Dict
    ) -> float:
        """Calculate quality reward based on output quality indicators."""
        steps = episode_data.get('steps', [])
        
        # Quality indicators from recent steps
        quality_indicators = []
        
        for step in steps[-3:]:  # Look at last 3 steps
            step_info = step.get('info', {})
            
            # Check for quality indicators
            if 'result' in step_info:
                result = step_info['result']
                if isinstance(result, dict):
                    # Quality indicators from search results
                    if 'results' in result:
                        avg_relevance = np.mean([r.get('relevance', 0.0) for r in result['results']])
                        quality_indicators.append(avg_relevance)
                    
                    # Quality indicators from content
                    if 'content' in result:
                        content_quality = min(len(result['content']) / 500.0, 1.0)
                        quality_indicators.append(content_quality)
                        
        # Average quality from indicators
        if quality_indicators:
            avg_quality = np.mean(quality_indicators)
        else:
            avg_quality = base_reward  # Fallback to base reward
            
        # Consistency bonus
        consistency_bonus = 0.0
        if len(quality_indicators) > 1:
            consistency = 1.0 - np.std(quality_indicators)
            consistency_bonus = max(0.0, consistency * 0.2)
            
        quality_reward = avg_quality + consistency_bonus
        
        return np.clip(quality_reward, 0.0, 1.0)
        
    def _calculate_exploration_reward(self, action: int, state: torch.Tensor, episode_data: Dict) -> float:
        """Calculate exploration reward to encourage diverse actions."""
        steps = episode_data.get('steps', [])
        
        # Action diversity
        recent_actions = [s.get('action', -1) for s in steps[-10:]]
        unique_actions = len(set(recent_actions))
        diversity_reward = unique_actions / 4.0  # Assuming 4 possible actions
        
        # State space exploration (simplified)
        state_norm = torch.norm(state).item()
        exploration_depth = min(state_norm / 10.0, 1.0)
        
        # Novelty bonus (decaying over time)
        novelty_bonus = self.current_exploration_bonus * (1.0 / (len(steps) + 1))
        
        exploration_reward = diversity_reward + exploration_depth + novelty_bonus
        
        return np.clip(exploration_reward, 0.0, 1.0)
        
    def _calculate_learning_reward(self, state: torch.Tensor, next_state: torch.Tensor, episode_data: Dict) -> float:
        """Calculate learning reward based on information gain."""
        # Information gain approximation
        state_change = torch.norm(next_state - state).item()
        information_gain = min(state_change / 5.0, 1.0)
        
        # Learning from mistakes (reward for recovery)
        steps = episode_data.get('steps', [])
        if len(steps) > 1:
            prev_reward = steps[-2].get('reward', 0.0)
            current_reward = steps[-1].get('reward', 0.0) if steps else 0.0
            
            if prev_reward < 0 and current_reward > prev_reward:
                recovery_bonus = 0.3  # Bonus for recovering from negative reward
            else:
                recovery_bonus = 0.0
        else:
            recovery_bonus = 0.0
            
        # Knowledge accumulation (simplified)
        knowledge_bonus = min(len(steps) * 0.05, 0.5)
        
        learning_reward = information_gain + recovery_bonus + knowledge_bonus
        
        return np.clip(learning_reward, 0.0, 1.0)
        
    def _calculate_coherence_reward(self, action: int, episode_data: Dict) -> float:
        """Calculate coherence reward based on action sequence logic."""
        steps = episode_data.get('steps', [])
        
        if len(steps) < 2:
            return 0.5  # Neutral for first actions
            
        # Action sequence coherence
        action_sequence = [s.get('action', -1) for s in steps[-5:]]
        
        # Define logical action transitions
        transition_scores = {
            (0, 1): 0.8,  # search -> browse
            (0, 2): 0.7,  # search -> reasoning
            (1, 2): 0.8,  # browse -> reasoning
            (2, 3): 0.9,  # reasoning -> synthesis
            (1, 3): 0.6,  # browse -> synthesis
            (0, 3): 0.4,  # search -> synthesis (less coherent)
        }
        
        # Calculate coherence score
        coherence_scores = []
        for i in range(1, len(action_sequence)):
            prev_action = action_sequence[i-1]
            curr_action = action_sequence[i]
            
            transition = (prev_action, curr_action)
            score = transition_scores.get(transition, 0.3)  # Default low coherence
            coherence_scores.append(score)
            
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        
        # Penalize excessive repetition
        repetition_penalty = 0.0
        if len(set(action_sequence)) == 1 and len(action_sequence) > 3:
            repetition_penalty = 0.3
            
        coherence_reward = avg_coherence - repetition_penalty
        
        return np.clip(coherence_reward, 0.0, 1.0)
        
    def _apply_global_smoothing(self, current_reward: float, base_reward: float) -> float:
        """Apply global smoothing to reduce reward variance."""
        if not self.global_reward_history:
            return current_reward
            
        # Get recent rewards
        recent_rewards = [h['reward'] for h in self.global_reward_history[-10:]]
        
        if len(recent_rewards) < 2:
            return current_reward
            
        # Calculate moving average
        moving_avg = np.mean(recent_rewards)
        
        # Smooth current reward towards moving average
        smoothing_factor = self.config.global_smoothing_factor
        smoothed_reward = (1 - smoothing_factor) * current_reward + smoothing_factor * moving_avg
        
        # Prevent over-smoothing (maintain some sensitivity to current performance)
        sensitivity_factor = 0.3
        final_reward = (1 - sensitivity_factor) * smoothed_reward + sensitivity_factor * current_reward
        
        return final_reward
        
    def _calculate_exploration_bonus(self, action: int, episode_data: Dict) -> float:
        """Calculate decaying exploration bonus."""
        steps = episode_data.get('steps', [])
        
        # Decay exploration bonus over time
        self.current_exploration_bonus *= self.exploration_bonus_decay
        
        # Action-specific exploration bonus
        action_counts = {}
        for step in steps:
            step_action = step.get('action', -1)
            action_counts[step_action] = action_counts.get(step_action, 0) + 1
            
        # Bonus for less-used actions
        current_action_count = action_counts.get(action, 0)
        max_action_count = max(action_counts.values()) if action_counts else 1
        
        if max_action_count > 0:
            rarity_bonus = (max_action_count - current_action_count) / max_action_count
        else:
            rarity_bonus = 0.0
            
        exploration_bonus = self.current_exploration_bonus * rarity_bonus
        
        return exploration_bonus
        
    def _apply_temporal_smoothing(self, reward: float) -> float:
        """Apply temporal smoothing to reduce sudden reward changes."""
        if len(self.global_reward_history) < 2:
            return reward
            
        # Get the last reward
        last_reward = self.global_reward_history[-1]['reward']
        
        # Limit the change between consecutive rewards
        max_change = self.config.max_reward_change
        
        reward_change = reward - last_reward
        if abs(reward_change) > max_change:
            # Clamp the change
            clamped_change = np.sign(reward_change) * max_change
            smoothed_reward = last_reward + clamped_change
        else:
            smoothed_reward = reward
            
        return smoothed_reward
        
    def get_category_stats(self) -> Dict[str, Dict]:
        """Get statistics for each reward category."""
        stats = {}
        
        for name, category in self.categories.items():
            stats[name] = {
                'weight': category.weight,
                'moving_average': category.get_smoothed_reward(),
                'recent_variance': category.get_recent_variance(),
                'history_length': len(category.history)
            }
            
        return stats
        
    def get_reward_history_stats(self) -> Dict[str, Any]:
        """Get overall reward history statistics."""
        if not self.global_reward_history:
            return {'total_rewards': 0}
            
        recent_rewards = [h['reward'] for h in self.global_reward_history[-50:]]
        
        return {
            'total_rewards': len(self.global_reward_history),
            'recent_average': np.mean(recent_rewards),
            'recent_std': np.std(recent_rewards),
            'recent_min': np.min(recent_rewards),
            'recent_max': np.max(recent_rewards),
            'exploration_bonus': self.current_exploration_bonus
        }
        
    def reset_episode(self):
        """Reset episode-specific state."""
        # Decay exploration bonus for new episode
        self.current_exploration_bonus *= self.exploration_bonus_decay
        
        # Store episode summary if there were rewards
        if self.global_reward_history:
            episode_rewards = [h['reward'] for h in self.global_reward_history[-100:]]
            self.episode_rewards.append({
                'episode_reward_sum': sum(episode_rewards),
                'episode_reward_avg': np.mean(episode_rewards),
                'timestamp': datetime.now()
            })
            
        self.logger.debug(f"Episode reset, exploration bonus: {self.current_exploration_bonus:.4f}")