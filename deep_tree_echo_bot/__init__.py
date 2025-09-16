"""
DeepTreeEchoBot - Enhanced AI agent with reinforcement learning capabilities.

This package implements an advanced deep-tree-echo-bot with:
- Strong Agentic Search
- Basic Browsing Capabilities  
- CPU-optimized Performance
- Focused Reasoning for Tasks
- Machine-generated Task Vectors
- Smooth Reward Functions
- Pure Reinforcement Learning (no supervised fine-tuning)
- Delta-Chat integration for messaging platform support
"""

__version__ = "0.1.0"
__author__ = "DeepTreeEchoBot Development Team"

from .core.bot import DeepTreeEchoBot
from .config.settings import BotConfig
from .rl.agent import RLAgent
from .search.agentic_search import AgenticSearchEngine
from .reasoning.task_processor import TaskProcessor
from .browsing.web_browser import WebBrowser
from .utils.task_vectors import TaskVectorGenerator
from .utils.reward_functions import SmoothRewardCalculator

# Delta-Chat integration (optional import)
try:
    from .deltachat_integration import DeltaChatBot, DeltaChatConfig
    __all__ = [
        'DeepTreeEchoBot',
        'BotConfig',
        'RLAgent',
        'AgenticSearchEngine', 
        'TaskProcessor',
        'WebBrowser',
        'TaskVectorGenerator',
        'SmoothRewardCalculator',
        'DeltaChatBot',
        'DeltaChatConfig'
    ]
except ImportError:
    # Delta-Chat integration not available
    __all__ = [
        'DeepTreeEchoBot',
        'BotConfig',
        'RLAgent',
        'AgenticSearchEngine', 
        'TaskProcessor',
        'WebBrowser',
        'TaskVectorGenerator',
        'SmoothRewardCalculator'
    ]
