"""
Command processor for Delta-Chat bot interactions.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass

from ..core.bot import DeepTreeEchoBot


@dataclass 
class CommandContext:
    """Context information for command execution."""
    command: str
    args: List[str]
    message_text: str
    sender_email: str
    chat_id: int
    is_group: bool
    is_admin: bool


class CommandProcessor:
    """Processes commands received via Delta-Chat."""
    
    def __init__(self, deep_tree_bot: DeepTreeEchoBot, command_prefix: str = "/"):
        self.deep_tree_bot = deep_tree_bot
        self.command_prefix = command_prefix
        self.logger = logging.getLogger(__name__)
        
        # Register built-in commands
        self.commands: Dict[str, Callable[[CommandContext], Awaitable[str]]] = {
            'help': self._cmd_help,
            'process': self._cmd_process,
            'search': self._cmd_search,
            'info': self._cmd_info,
            'status': self._cmd_status,
            'stats': self._cmd_stats,
            'ping': self._cmd_ping,
        }
        
    def register_command(self, name: str, handler: Callable[[CommandContext], Awaitable[str]]):
        """Register a custom command handler."""
        self.commands[name] = handler
        
    def is_command(self, message: str) -> bool:
        """Check if message is a command."""
        return message.strip().startswith(self.command_prefix)
        
    def parse_command(self, message: str) -> tuple[str, List[str]]:
        """Parse command and arguments from message."""
        message = message.strip()
        if not message.startswith(self.command_prefix):
            return "", []
            
        # Remove prefix and split
        parts = message[len(self.command_prefix):].split()
        if not parts:
            return "", []
            
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        return command, args
        
    async def process_command(self, context: CommandContext) -> str:
        """Process a command and return response."""
        command = context.command.lower()
        
        if command not in self.commands:
            return f"Unknown command: {command}. Type {self.command_prefix}help for available commands."
            
        try:
            self.logger.info(f"Processing command '{command}' from {context.sender_email}")
            response = await self.commands[command](context)
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing command '{command}': {e}")
            return f"Error processing command: {str(e)}"
            
    async def _cmd_help(self, context: CommandContext) -> str:
        """Show available commands."""
        commands = []
        
        # Basic commands
        basic_commands = ['help', 'process', 'search', 'info', 'status', 'ping']
        commands.append("**Available Commands:**")
        for cmd in basic_commands:
            if cmd in self.commands:
                commands.append(f"{self.command_prefix}{cmd} - {self._get_command_description(cmd)}")
                
        # Admin commands (if user is admin)
        if context.is_admin:
            admin_commands = ['stats']
            if admin_commands:
                commands.append("\n**Admin Commands:**")
                for cmd in admin_commands:
                    if cmd in self.commands:
                        commands.append(f"{self.command_prefix}{cmd} - {self._get_command_description(cmd)}")
                        
        return "\n".join(commands)
        
    def _get_command_description(self, command: str) -> str:
        """Get description for a command."""
        descriptions = {
            'help': 'Show this help message',
            'process': 'Process a task with the AI bot',
            'search': 'Search for information',
            'info': 'Show bot information',
            'status': 'Show bot status',
            'stats': 'Show bot statistics (admin only)',
            'ping': 'Check if bot is responsive'
        }
        return descriptions.get(command, 'No description available')
        
    async def _cmd_process(self, context: CommandContext) -> str:
        """Process a task using the DeepTreeEchoBot."""
        if not context.args:
            return f"Usage: {self.command_prefix}process <task description>"
            
        task = " ".join(context.args)
        
        try:
            # Add timeout to prevent long-running tasks from blocking
            result = await asyncio.wait_for(
                self.deep_tree_bot.process_task(task),
                timeout=30.0
            )
            
            if result['success']:
                response = f"✅ **Task completed**\n"
                response += f"**Reward:** {result['total_reward']:.3f}\n"
                response += f"**Steps:** {result['steps_taken']}\n"
                if result.get('result'):
                    # Truncate long results
                    result_text = str(result['result'])
                    if len(result_text) > 1500:
                        result_text = result_text[:1500] + "..."
                    response += f"**Result:** {result_text}"
                return response
            else:
                return f"❌ Task failed to complete. Steps taken: {result['steps_taken']}"
                
        except asyncio.TimeoutError:
            return "⏱️ Task processing timed out. Please try a simpler task."
        except Exception as e:
            return f"❌ Error processing task: {str(e)}"
            
    async def _cmd_search(self, context: CommandContext) -> str:
        """Search for information."""
        if not context.args:
            return f"Usage: {self.command_prefix}search <search query>"
            
        query = " ".join(context.args)
        
        try:
            # Use the search engine component directly for faster response
            search_results = await self.deep_tree_bot.search_engine.search(query)
            
            if not search_results.get('results'):
                return f"No search results found for: {query}"
                
            response = f"🔍 **Search results for:** {query}\n\n"
            
            # Show top 3 results
            for i, result in enumerate(search_results['results'][:3], 1):
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No description')
                url = result.get('url', '')
                
                response += f"**{i}. {title}**\n"
                if snippet:
                    # Truncate long snippets
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    response += f"{snippet}\n"
                if url:
                    response += f"{url}\n"
                response += "\n"
                
            return response.strip()
            
        except Exception as e:
            return f"❌ Search error: {str(e)}"
            
    async def _cmd_info(self, context: CommandContext) -> str:
        """Show bot information."""
        return (
            "🤖 **DeepTreeEchoBot** - Enhanced AI Agent\n\n"
            "**Features:**\n"
            "• Reinforcement Learning-based task processing\n"
            "• Intelligent search capabilities\n"
            "• Web browsing and content analysis\n"
            "• Focused reasoning for complex tasks\n\n"
            f"**Commands:** Type {self.command_prefix}help for available commands\n"
            "**Version:** 0.1.0"
        )
        
    async def _cmd_status(self, context: CommandContext) -> str:
        """Show bot status."""
        try:
            rl_stats = self.deep_tree_bot.rl_agent.get_training_stats()
            episode_count = len(self.deep_tree_bot.episode_history)
            
            return (
                "📊 **Bot Status**\n\n"
                f"**Running:** ✅ Active\n"
                f"**Episodes processed:** {episode_count}\n"
                f"**Training steps:** {rl_stats.get('training_step', 0)}\n"
                f"**Current epsilon:** {rl_stats.get('epsilon', 0):.3f}\n"
                f"**Buffer size:** {rl_stats.get('buffer_size', 0)}"
            )
        except Exception as e:
            return f"Status check failed: {str(e)}"
            
    async def _cmd_stats(self, context: CommandContext) -> str:
        """Show detailed bot statistics (admin only)."""
        if not context.is_admin:
            return "❌ Admin access required for this command."
            
        try:
            rl_stats = self.deep_tree_bot.rl_agent.get_training_stats()
            episodes = self.deep_tree_bot.episode_history
            
            response = "📈 **Detailed Bot Statistics**\n\n"
            response += f"**Training Stats:**\n"
            response += f"• Steps: {rl_stats.get('training_step', 0)}\n"
            response += f"• Episodes: {rl_stats.get('episode_count', 0)}\n"
            response += f"• Epsilon: {rl_stats.get('epsilon', 0):.4f}\n"
            response += f"• Buffer: {rl_stats.get('buffer_size', 0)}\n\n"
            
            if episodes:
                recent_rewards = [ep['total_reward'] for ep in episodes[-10:]]
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                response += f"**Recent Performance:**\n"
                response += f"• Average reward (last 10): {avg_reward:.3f}\n"
                response += f"• Best reward: {max(ep['total_reward'] for ep in episodes):.3f}\n"
                
            return response
            
        except Exception as e:
            return f"❌ Error getting statistics: {str(e)}"
            
    async def _cmd_ping(self, context: CommandContext) -> str:
        """Simple ping command."""
        return "🏓 Pong! Bot is responsive."