"""
Configuration settings for DeepTreeEchoBot.
Centralized configuration management for all components.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import os


@dataclass
class RLConfig:
    """Reinforcement Learning configuration."""
    state_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    gamma: float = 0.99
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100


@dataclass
class SearchConfig:
    """Search engine configuration."""
    context_feature_dim: int = 16
    max_results_per_engine: int = 10
    search_timeout: int = 30
    enable_semantic_ranking: bool = True
    cache_results: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class ReasoningConfig:
    """Task processor/reasoning configuration."""
    state_feature_dim: int = 32
    max_reasoning_steps: int = 10
    step_timeout: int = 30
    enable_task_decomposition: bool = True
    context_memory_size: int = 1000


@dataclass
class BrowsingConfig:
    """Web browsing configuration."""
    request_timeout: int = 30
    max_page_size: int = 1024 * 1024  # 1MB
    max_content_length: int = 10000
    max_links_per_page: int = 20
    enable_javascript: bool = False
    user_agent: str = "DeepTreeEchoBot/1.0"


@dataclass
class TaskVectorConfig:
    """Task vector generation configuration."""
    vector_dimension: int = 64
    learning_rate: float = 0.1
    max_history_size: int = 1000
    enable_domain_adaptation: bool = True
    complexity_scaling: bool = True


@dataclass
class RewardConfig:
    """Reward function configuration."""
    global_smoothing_factor: float = 0.1
    max_reward_change: float = 0.5
    max_history_size: int = 1000
    initial_exploration_bonus: float = 0.2
    enable_category_weighting: bool = True


@dataclass
class BotConfig:
    """Main bot configuration combining all components."""
    # Component configurations
    rl_config: RLConfig = field(default_factory=RLConfig)
    search_config: SearchConfig = field(default_factory=SearchConfig)
    reasoning_config: ReasoningConfig = field(default_factory=ReasoningConfig)
    browsing_config: BrowsingConfig = field(default_factory=BrowsingConfig)
    task_vector_config: TaskVectorConfig = field(default_factory=TaskVectorConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    
    # Global settings
    max_steps_per_task: int = 50
    action_feature_dim: int = 8
    cpu_threads: int = 4
    memory_limit_mb: int = 512
    
    # Logging and debugging
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    save_episode_data: bool = True
    
    # Model persistence
    model_save_path: str = "models/"
    checkpoint_frequency: int = 100  # episodes
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BotConfig':
        """Create config from dictionary."""
        # Extract component configs
        rl_config = RLConfig(**config_dict.get('rl', {}))
        search_config = SearchConfig(**config_dict.get('search', {}))
        reasoning_config = ReasoningConfig(**config_dict.get('reasoning', {}))
        browsing_config = BrowsingConfig(**config_dict.get('browsing', {}))
        task_vector_config = TaskVectorConfig(**config_dict.get('task_vectors', {}))
        reward_config = RewardConfig(**config_dict.get('rewards', {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['rl', 'search', 'reasoning', 'browsing', 'task_vectors', 'rewards']}
        
        return cls(
            rl_config=rl_config,
            search_config=search_config,
            reasoning_config=reasoning_config,
            browsing_config=browsing_config,
            task_vector_config=task_vector_config,
            reward_config=reward_config,
            **main_config
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'rl': {
                'state_dim': self.rl_config.state_dim,
                'hidden_dims': self.rl_config.hidden_dims,
                'learning_rate': self.rl_config.learning_rate,
                'gamma': self.rl_config.gamma,
                'initial_epsilon': self.rl_config.initial_epsilon,
                'epsilon_decay': self.rl_config.epsilon_decay,
                'epsilon_min': self.rl_config.epsilon_min,
                'buffer_size': self.rl_config.buffer_size,
                'batch_size': self.rl_config.batch_size,
                'target_update_freq': self.rl_config.target_update_freq,
            },
            'search': {
                'context_feature_dim': self.search_config.context_feature_dim,
                'max_results_per_engine': self.search_config.max_results_per_engine,
                'search_timeout': self.search_config.search_timeout,
                'enable_semantic_ranking': self.search_config.enable_semantic_ranking,
                'cache_results': self.search_config.cache_results,
                'cache_ttl': self.search_config.cache_ttl,
            },
            'reasoning': {
                'state_feature_dim': self.reasoning_config.state_feature_dim,
                'max_reasoning_steps': self.reasoning_config.max_reasoning_steps,
                'step_timeout': self.reasoning_config.step_timeout,
                'enable_task_decomposition': self.reasoning_config.enable_task_decomposition,
                'context_memory_size': self.reasoning_config.context_memory_size,
            },
            'browsing': {
                'request_timeout': self.browsing_config.request_timeout,
                'max_page_size': self.browsing_config.max_page_size,
                'max_content_length': self.browsing_config.max_content_length,
                'max_links_per_page': self.browsing_config.max_links_per_page,
                'enable_javascript': self.browsing_config.enable_javascript,
                'user_agent': self.browsing_config.user_agent,
            },
            'task_vectors': {
                'vector_dimension': self.task_vector_config.vector_dimension,
                'learning_rate': self.task_vector_config.learning_rate,
                'max_history_size': self.task_vector_config.max_history_size,
                'enable_domain_adaptation': self.task_vector_config.enable_domain_adaptation,
                'complexity_scaling': self.task_vector_config.complexity_scaling,
            },
            'rewards': {
                'global_smoothing_factor': self.reward_config.global_smoothing_factor,
                'max_reward_change': self.reward_config.max_reward_change,
                'max_history_size': self.reward_config.max_history_size,
                'initial_exploration_bonus': self.reward_config.initial_exploration_bonus,
                'enable_category_weighting': self.reward_config.enable_category_weighting,
            },
            # Global settings
            'max_steps_per_task': self.max_steps_per_task,
            'action_feature_dim': self.action_feature_dim,
            'cpu_threads': self.cpu_threads,
            'memory_limit_mb': self.memory_limit_mb,
            'log_level': self.log_level,
            'enable_debug_mode': self.enable_debug_mode,
            'save_episode_data': self.save_episode_data,
            'model_save_path': self.model_save_path,
            'checkpoint_frequency': self.checkpoint_frequency,
        }
        
    @classmethod
    def get_cpu_optimized_config(cls) -> 'BotConfig':
        """Get a configuration optimized for CPU performance."""
        config = cls()
        
        # Optimize RL config for CPU
        config.rl_config.hidden_dims = [128, 64]  # Smaller networks
        config.rl_config.batch_size = 16  # Smaller batches
        config.rl_config.buffer_size = 5000  # Smaller buffer
        
        # Optimize search config
        config.search_config.max_results_per_engine = 5
        config.search_config.search_timeout = 15
        
        # Optimize browsing config
        config.browsing_config.max_page_size = 512 * 1024  # 512KB
        config.browsing_config.max_content_length = 5000
        config.browsing_config.max_links_per_page = 10
        
        # Optimize task vectors
        config.task_vector_config.vector_dimension = 32  # Smaller vectors
        config.task_vector_config.max_history_size = 500
        
        # Global CPU optimizations
        config.cpu_threads = min(4, os.cpu_count() or 1)
        config.memory_limit_mb = 256
        config.max_steps_per_task = 25
        
        return config
        
    @classmethod
    def get_development_config(cls) -> 'BotConfig':
        """Get a configuration suitable for development and testing."""
        config = cls.get_cpu_optimized_config()
        
        # Development-specific settings
        config.log_level = "DEBUG"
        config.enable_debug_mode = True
        config.save_episode_data = True
        
        # Faster training for development
        config.rl_config.initial_epsilon = 0.5
        config.rl_config.epsilon_decay = 0.99
        
        # More frequent checkpoints
        config.checkpoint_frequency = 10
        
        return config