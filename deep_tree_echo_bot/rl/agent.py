"""
Reinforcement Learning Agent for DeepTreeEchoBot.
Pure RL implementation without supervised fine-tuning.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple, List, Optional
import logging

from ..config.settings import RLConfig


class DQNetwork(nn.Module):
    """Deep Q-Network for action selection."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(
        self, 
        state: torch.Tensor, 
        action: int, 
        reward: float, 
        next_state: torch.Tensor, 
        done: bool
    ):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([t[0] for t in batch])
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.stack([t[3] for t in batch])
        dones = torch.tensor([t[4] for t in batch], dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)


class RLAgent:
    """
    Pure Reinforcement Learning Agent using Deep Q-Learning.
    No supervised fine-tuning - learns purely from environment interactions.
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Action space
        self.action_space = {
            0: 'search',
            1: 'browse', 
            2: 'reasoning',
            3: 'synthesis'
        }
        self.num_actions = len(self.action_space)
        
        # Networks
        self.q_network = DQNetwork(
            state_dim=config.state_dim,
            action_dim=self.num_actions,
            hidden_dims=config.hidden_dims
        )
        
        self.target_network = DQNetwork(
            state_dim=config.state_dim,
            action_dim=self.num_actions,
            hidden_dims=config.hidden_dims
        )
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training parameters
        self.epsilon = config.initial_epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        
    async def initialize(self):
        """Initialize the RL agent."""
        self.logger.info("Initializing RL Agent...")
        
        # Set networks to training mode
        self.q_network.train()
        self.target_network.eval()
        
        self.logger.info(f"RL Agent initialized with {self.num_actions} actions")
        
    async def get_action(self, state: torch.Tensor, task_vector: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
            task_vector: Task-specific vector for context
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
            self.logger.debug(f"Random action selected: {action} ({self.action_space[action]})")
        else:
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                action = q_values.argmax().item()
                self.logger.debug(f"Greedy action selected: {action} ({self.action_space[action]})")
                
        return action
        
    async def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    async def train_step(self):
        """Perform one training step if enough samples are available."""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update training step
        self.training_step += 1
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.logger.debug("Target network updated")
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.logger.debug(f"Training step {self.training_step}, loss: {loss.item():.4f}, epsilon: {self.epsilon:.4f}")
        
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
        }, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        
        self.logger.info(f"Model loaded from {path}")
        
    def get_training_stats(self) -> dict:
        """Get current training statistics."""
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'action_space': self.action_space
        }